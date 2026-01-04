import os

import gymnasium as gym
import gymnasium_robotics
import hydra
import jax
import numpy as np
import omegaconf
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from flowrl.agent.online import *
from flowrl.config.online.mujoco_config import Config
from flowrl.dataset.buffer.state import ReplayBuffer
from flowrl.types import *
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

jax.config.update("jax_default_matmul_precision", "float32")

SUPPORTED_AGENTS: Dict[str, BaseAgent] = {
    "sac": SACAgent,
    "td3": TD3Agent,
    "td7": TD7Agent,
    "sdac": SDACAgent,
    "dpmd": DPMDAgent,
}

class OffPolicyTrainer():
    def __init__(self, cfg: Config):
        self.cfg = cfg

        set_seed_everywhere(cfg.seed)
        self.logger = CompositeLogger(
            log_dir="/".join([cfg.log.dir, cfg.algo.name, cfg.log.tag, cfg.task]),
            name="seed"+str(cfg.seed),
            logger_config={
                "TensorboardLogger": {"activate": True},
                "WandbLogger": {
                    "activate": True,
                    "config": OmegaConf.to_container(cfg),
                    "settings": wandb.Settings(_disable_stats=True),
                    "project": cfg.log.project,
                    "entity": cfg.log.entity
                } if ("project" in cfg.log and "entity" in cfg.log) else {"activate": False},
            }
        )
        self.ckpt_save_dir = os.path.join(self.logger.log_dir, "ckpt")
        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))
        print("="*35+" Config "+"="*35)
        print(OmegaConf.to_yaml(cfg))
        print("="*80)
        print(f"\nSave results to: {self.logger.log_dir}\n")

        # create env
        assert cfg.train_frames % (cfg.frame_skip * cfg.num_train_envs) == 0, f"train_frames ({cfg.train_frames}) must be divisible by frame_skip ({cfg.frame_skip}) * num_train_envs ({cfg.num_train_envs})"
        assert cfg.log_frames % (cfg.frame_skip * cfg.num_train_envs) == 0, f"log_frames ({cfg.log_frames}) must be divisible by frame_skip ({cfg.frame_skip}) * num_train_envs ({cfg.num_train_envs})"
        assert int(cfg.num_train_envs * cfg.utd) > 0, f"num_train_envs ({cfg.num_train_envs}) * utd ({cfg.utd}) must be greater than 0"
        self.frame_skip = cfg.frame_skip
        self.num_train_envs = cfg.num_train_envs
        self.update_per_iter = int(cfg.num_train_envs * cfg.utd)
        self.train_env = gym.vector.SyncVectorEnv([
            lambda: gym.make(cfg.task) for _ in range(cfg.num_train_envs)
        ], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        self.eval_env = gym.vector.SyncVectorEnv([
            lambda: gym.make(cfg.task) for _ in range(cfg.eval.num_episodes)
        ], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

        # create buffer
        self.use_lap_buffer = cfg.lap_alpha > 0
        self.buffer = ReplayBuffer(
            obs_dim=self.train_env.observation_space.shape[-1],
            action_dim=self.train_env.action_space.shape[-1],
            max_size=cfg.buffer_size,
            norm_obs=cfg.norm_obs,
            norm_reward=cfg.norm_reward,
            lap_alpha=cfg.lap_alpha,
        )

        # create agent
        self.agent = SUPPORTED_AGENTS[cfg.algo.name](
            obs_dim=self.train_env.observation_space.shape[-1],
            act_dim=self.train_env.action_space.shape[-1],
            cfg=cfg.algo,
            seed=cfg.seed,
        )

        self.global_step = 0

    @property
    def global_frame(self) -> int:
        return self.frame_skip * self.global_step

    def train(self):
        cfg = self.cfg
        try:
            obs, _ = self.train_env.reset()
            with tqdm(total=cfg.train_frames, desc="training") as pbar:
                while self.global_frame < cfg.train_frames:
                    actions, _ = self.agent.sample_actions(
                        self.buffer.normalize_obs(obs),
                        deterministic=False,
                        num_samples=1
                    )
                    if self.global_frame < cfg.random_frames:
                        actions = self.train_env.action_space.sample()

                    next_obs, rewards, terminated, truncated, infos = self.train_env.step(actions)

                    for i in range(self.num_train_envs):
                        # get the actual next observation
                        if terminated[i] or truncated[i]:
                            actual_next_obs = infos["final_obs"][i]
                        else:
                            actual_next_obs = next_obs[i]
                        self.buffer.add(obs[i], actions[i], actual_next_obs, rewards[i], terminated[i])

                    if self.global_frame < cfg.warmup_frames:
                        train_metrics = {}
                    else:
                        for _ in range(self.update_per_iter):
                            batch, indices = self.buffer.sample(batch_size=cfg.batch_size)
                            train_metrics = self.agent.train_step(batch, step=self.global_frame)
                            if self.use_lap_buffer:
                                new_priorities = train_metrics.pop("priority")
                                self.buffer.update(indices, new_priorities)
                                train_metrics["misc/max_priority"] = self.buffer.max_priority

                    if self.use_lap_buffer and self.global_frame % cfg.lap_reset_frames == 0:
                        self.buffer.reset_max_priority()

                    if self.global_frame % cfg.log_frames == 0:
                        self.logger.log_scalars("", train_metrics, step=self.global_frame)

                    if self.global_frame % cfg.eval_frames == 0:
                        self.eval_and_save()

                    # update step count and obs
                    self.global_step += self.num_train_envs
                    obs = next_obs
                    pbar.update(self.num_train_envs)
                self.eval_and_save()

        except KeyboardInterrupt:
            print("Stopped by keyboard interruption. ")
        except Exception as e:
            raise e

    def eval_and_save(self):
        # initialize arrays to store results
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)

        # reset all environments
        obs, _ = self.eval_env.reset()
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)
        dones = np.zeros(self.cfg.eval.num_episodes, dtype=bool)

        # run episodes in parallel
        while not np.all(dones):
            # get actions for all environments
            actions, _ = self.agent.sample_actions(
                obs,
                deterministic=True,
                num_samples=self.cfg.eval.num_samples,
            )

            # step all environments
            obs, rewards, terminated, truncated, infos = self.eval_env.step(actions)
            new_dones = terminated | truncated

            returns += rewards * (1-dones)
            lengths += 1 * (1-dones)
            dones = dones | new_dones

        eval_metrics = {
            "mean": np.mean(returns),
            "median": np.median(returns),
            "std": np.std(returns),
            "min": np.min(returns),
            "max": np.max(returns),
            "length": np.mean(lengths),
        }
        self.logger.log_scalars("eval", eval_metrics, step=self.global_frame)
        if self.cfg.log.save_ckpt:
            self.agent.save(os.path.join(self.ckpt_save_dir, f"{self.global_frame}"))


class OnPolicyTrainer():
    pass


@hydra.main(config_path="./config/mujoco", config_name="config", version_base=None)
def main(cfg: Config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    try:
        algo_name = cfg.algo.name
    except omegaconf.errors.MissingMandatoryValue:
        err_string = "Algorithm is not specified. Please specify the algorithm via `algo=<algo_name>` in command."
        err_string += "\nAvailable algorithms are:\n  "
        err_string += "\n  ".join(SUPPORTED_AGENTS.keys())
        print(err_string)
        exit(1)

    trainer = OffPolicyTrainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()
