import os
import warnings

os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
warnings.filterwarnings("ignore", message=".*DISPLAY environment variable.*")

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
from omegaconf import OmegaConf
from tqdm import tqdm, trange

import wandb
from flowrl.agent.online import *
from flowrl.config.online.hb_config import Config
from flowrl.dataset.buffer.state import ReplayBuffer, RMSNormalizer, BatchedRewardNormalizer
from flowrl.env.online.humanoidbench_env import HumanoidBenchEnv
from flowrl.types import *
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

jax.config.update("jax_default_matmul_precision", "float32")

SUPPORTED_AGENTS: Dict[str, BaseAgent] = {
    "td3": TD3Agent,
    "diffsr_td3": DiffSRTD3Agent,
    "brc": BRCAgent,
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

        self.frame_skip = cfg.frame_skip
        self.num_train_envs = cfg.num_train_envs
        self.update_per_iter = int(self.num_train_envs * cfg.utd)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)

        print(f"Creating environments for task: {cfg.task} (num_train_envs={self.num_train_envs})")
        self.train_env = HumanoidBenchEnv(
            task=cfg.task, num_envs=self.num_train_envs, seed=cfg.seed,
            frame_skip=cfg.frame_skip, frame_stack=cfg.frame_stack
        )
        self.eval_env = HumanoidBenchEnv(
            task=cfg.task, num_envs=cfg.eval.num_episodes, seed=cfg.seed + 1000,
            frame_skip=cfg.frame_skip, frame_stack=cfg.frame_stack
        )

        self.use_lap_buffer = cfg.lap_alpha > 0
        self.obs_dim = self.train_env.observation_space.shape[-1]
        self.action_dim = self.train_env.action_space.shape[-1]
        self.buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            max_size=cfg.buffer_size,
            lap_alpha=cfg.lap_alpha,
        )
        if cfg.norm_obs:
            self.obs_normalizer = RMSNormalizer(shape=(self.obs_dim,))
        if cfg.norm_reward:
            self.reward_normalizer = BatchedRewardNormalizer(
                num_envs=self.num_train_envs,
                discount=cfg.discount,
                v_max=10.0,
                target_entropy=-self.action_dim / 2,
            )

        # create agent
        self.agent = SUPPORTED_AGENTS[cfg.algo.name](
            obs_dim=self.obs_dim,
            act_dim=self.action_dim,
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
        ep_lengths = np.zeros(self.num_train_envs)
        ep_returns = np.zeros(self.num_train_envs)

        obs, _ = self.train_env.reset()
        with tqdm(total=cfg.train_frames, desc="training") as pbar:
            while self.global_frame <= cfg.train_frames:
                actions, _ = self.agent.sample_actions(
                    self.obs_normalizer.normalize(obs) if self.cfg.norm_obs else obs,
                    deterministic=False, num_samples=1,
                )
                if self.global_frame < cfg.random_frames:
                    actions = self.train_env.action_space.sample()

                next_obs, rewards, terminated, truncated, infos = self.train_env.step(actions)

                for i in range(self.num_train_envs):
                    ep_lengths[i] += 1
                    ep_returns[i] += rewards[i]

                    if terminated[i] or truncated[i]:
                        actual_next_obs = infos["final_obs"][i]
                    else:
                        actual_next_obs = next_obs[i]

                    self.buffer.add(obs[i], actions[i], actual_next_obs, rewards[i], terminated[i])

                    if self.cfg.norm_obs:
                        self.obs_normalizer.update(obs[i])
                    if self.cfg.norm_reward:
                        self.reward_normalizer.update(i, rewards[i], terminated[i], truncated[i])

                    if terminated[i] or truncated[i]:
                        self.global_episode += 1
                        self.logger.log_scalars("", {
                            "rollout/episode_return": ep_returns[i],
                            "rollout/episode_length": ep_lengths[i]
                        }, step=self.global_frame)
                        ep_lengths[i] = ep_returns[i] = 0

                if self.global_frame < cfg.warmup_frames:
                    train_metrics = {}
                else:
                    for _ in range(self.update_per_iter):
                        batch, indices = self.buffer.sample(batch_size=cfg.batch_size)
                        if self.cfg.norm_obs:
                            batch.obs = self.obs_normalizer.normalize(batch.obs)
                            batch.next_obs = self.obs_normalizer.normalize(batch.next_obs)
                        if self.cfg.norm_reward:
                            batch.reward = self.reward_normalizer.normalize(batch.reward, temperature=jnp.exp(self.agent.log_alpha()))
                        train_metrics = self.agent.train_step(batch, step=self.global_frame, num_updates=1)
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

                self.global_step += self.num_train_envs
                obs = next_obs
                pbar.update(self.frame_skip * self.num_train_envs)

    def eval_and_save(self):
        # initialize arrays to store results
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)
        dones = np.zeros(self.cfg.eval.num_episodes, dtype=bool)

        # reset all environments
        obs, _ = self.eval_env.reset()
        while not np.all(dones):
            actions, _ = self.agent.sample_actions(
                self.obs_normalizer.normalize(obs) if self.cfg.norm_obs else obs,
                deterministic=True, num_samples=self.cfg.eval.num_samples,
            )
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


@hydra.main(config_path="./config/hb", config_name="config", version_base=None)
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
