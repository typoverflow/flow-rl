import os

import gymnasium as gym
import hydra
import jax.numpy as jnp
import numpy as np
import omegaconf
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from flowrl.agent.online import *
from flowrl.config.online.mujoco import Config
from flowrl.dataset.buffer.state import ReplayBuffer
from flowrl.env.online.dmc_env import DMControlEnv
from flowrl.types import *
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

SUPPORTED_AGENTS: Dict[str, BaseAgent] = {
    "sac": SACAgent,
    "td3": TD3Agent,
    "td7": TD7Agent,
    "sdac": SDACAgent,
    "dpmd": DPMDAgent,
    "ctrl_td3": Ctrl_TD3_Agent,
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
        self.frame_skip = cfg.frame_skip
        self.train_env = DMControlEnv(cfg.task, cfg.seed, False, cfg.frame_skip, cfg.frame_stack)
        self.eval_env = [DMControlEnv(cfg.task, cfg.seed + i*100, False, cfg.frame_skip, cfg.frame_stack) for i in range(cfg.eval.num_episodes)]

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
        self.global_episode = 0

    @property
    def global_frame(self) -> int:
        return self.frame_skip * self.global_step

    def train(self):
        cfg = self.cfg

        ep_length = ep_return = 0
        obs, _ = self.train_env.reset()
        with tqdm(total=cfg.train_frames, desc="training") as pbar:
            while self.global_frame <= cfg.train_frames:
                action, _ = self.agent.sample_actions(
                    self.buffer.normalize_obs(obs[jnp.newaxis, ...]),
                    deterministic=False,
                    num_samples=1,
                )
                action = np.asarray(action[0])
                if self.global_frame < cfg.random_frames:
                    action = self.train_env.action_space.sample()

                next_obs, reward, terminated, truncated, info = self.train_env.step(action)
                ep_length += 1
                ep_return += reward

                self.buffer.add(obs, action, next_obs, reward, terminated)

                if terminated or truncated:
                    next_obs, _ = self.train_env.reset()
                    self.global_episode += 1
                    self.logger.log_scalars("", {
                        "rollout/episode_return": ep_return,
                        "rollout/episode_length": ep_length
                    }, step=self.global_frame)
                    ep_length = ep_return = 0

                if self.global_frame < cfg.warmup_frames:
                    train_metrics = {}
                else:
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

                self.global_step += 1
                obs = next_obs
                pbar.update(self.frame_skip)

    def eval_and_save(self):
        # initialize arrays to store results
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)

        # reset all environments
        obs = np.stack([env.reset()[0] for env in self.eval_env], axis=0)
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)
        dones = np.zeros(self.cfg.eval.num_episodes, dtype=bool)

        # run episodes in parallel
        while not np.all(dones):
            # get actions for all environments
            actions, _ = self.agent.sample_actions(
                self.buffer.normalize_obs(obs),
                deterministic=True,
                num_samples=self.cfg.eval.num_samples,
            )
            actions = np.asarray(actions)

            # step all environments
            obs, rewards, terminated, truncated, infos = zip(*[env.step(action) for env, action in zip(self.eval_env, actions)])
            obs = np.stack(obs, axis=0)
            rewards = np.stack(rewards, axis=0)
            terminated = np.stack(terminated, axis=0)
            truncated = np.stack(truncated, axis=0)
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


@hydra.main(config_path="./config/dmc", config_name="config", version_base=None)
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
