import os

os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
from omegaconf import OmegaConf
from tqdm import tqdm

import wandb
from flowrl.agent.online.ppo import PPOAgent
from flowrl.config.online.onpolicy_hb_config import Config
from flowrl.dataset.buffer.state import RMSNormalizer
from flowrl.env.online.humanoidbench_env import HumanoidBenchEnv
from flowrl.types import Dict, RolloutBatch
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

jax.config.update("jax_default_matmul_precision", "float32")

SUPPORTED_AGENTS: Dict[str, type] = {
    "ppo": PPOAgent,
}


class HumanoidBenchOnPolicyTrainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

        set_seed_everywhere(cfg.seed)
        self.logger = CompositeLogger(
            log_dir="/".join([cfg.log.dir, cfg.algo.name, cfg.log.tag, cfg.task]),
            name="seed" + str(cfg.seed),
            logger_config={
                "TensorboardLogger": {"activate": True},
                "WandbLogger": {
                    "activate": True,
                    "config": OmegaConf.to_container(cfg),
                    "settings": wandb.Settings(_disable_stats=True),
                    "project": cfg.log.project,
                    "entity": cfg.log.entity,
                } if ("project" in cfg.log and "entity" in cfg.log) else {"activate": False},
            },
        )
        self.ckpt_save_dir = os.path.join(self.logger.log_dir, "ckpt")
        OmegaConf.save(cfg, os.path.join(self.logger.log_dir, "config.yaml"))
        print("=" * 35 + " Config " + "=" * 35)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)
        print(f"\nSave results to: {self.logger.log_dir}\n")

        # Create vectorized training envs (like DMC on-policy)
        self.num_envs = cfg.num_envs
        self.rollout_length = cfg.rollout_length
        self.train_env = gym.vector.SyncVectorEnv([
            lambda: gym.wrappers.RescaleAction(
                HumanoidBenchEnv(
                    cfg.task,
                    cfg.seed + i,
                    cfg.frame_skip,
                    cfg.frame_stack,
                ),
                min_action=-1.0,
                max_action=1.0,
            )
            for i in range(self.num_envs)
        ], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        self.eval_env = gym.vector.SyncVectorEnv([
            lambda: gym.wrappers.RescaleAction(
                HumanoidBenchEnv(
                    cfg.task,
                    cfg.seed + 10000 + i,
                    cfg.frame_skip,
                    cfg.frame_stack
                ),
                min_action=-1.0,
                max_action=1.0,
            )
            for i in range(cfg.eval.num_episodes)
        ], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

        self.obs_dim = self.train_env.observation_space.shape[-1]
        self.action_dim = self.train_env.action_space.shape[-1]
        self.frame_skip = cfg.frame_skip

        if cfg.norm_obs:
            self.obs_normalizer = RMSNormalizer(shape=(self.obs_dim,))

        # Create agent
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

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.cfg.norm_obs:
            return self.obs_normalizer.normalize(obs)
        return obs

    def collect_rollouts(self) -> RolloutBatch:
        """Collect rollouts from vectorized HumanoidBench envs."""
        T = self.rollout_length
        B = self.num_envs

        all_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_actions = np.zeros((T, B, self.action_dim), dtype=np.float32)
        all_rewards = np.zeros((T, B, 1), dtype=np.float32)
        all_terminated = np.zeros((T, B, 1), dtype=np.float32)
        all_truncated = np.zeros((T, B, 1), dtype=np.float32)
        all_log_probs = np.zeros((T, B, 1), dtype=np.float32)
        last_obs = np.zeros((B, self.obs_dim), dtype=np.float32)

        for t in range(T):
            if self.cfg.norm_obs:
                self.obs_normalizer.update(self.obs)
            obs_norm = self._normalize_obs(self.obs)
            all_obs[t] = obs_norm

            actions, info = self.agent.sample_actions(
                jnp.array(obs_norm), deterministic=False
            )
            actions_clipped = np.clip(actions, -1.0, 1.0)
            all_actions[t] = actions
            all_log_probs[t] = info["log_prob"]

            next_obs, rewards, terminated, truncated, infos = self.train_env.step(actions_clipped)
            all_rewards[t] = rewards[..., jnp.newaxis]
            last_obs = next_obs

            self.ep_returns += rewards
            self.ep_lengths += 1

            done = terminated | truncated
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
                all_terminated[t, terminated] = 1.0
                all_truncated[t, truncated] = 1.0
                last_obs[done_indices] = np.stack(infos["final_obs"][done_indices], axis=0)

                # log completed episode stats
                mean_return = np.mean(self.ep_returns[done_indices])
                mean_length = np.mean(self.ep_lengths[done_indices])
                self.logger.log_scalars("", {
                    "rollout/episode_return": mean_return,
                    "rollout/episode_length": mean_length,
                    "rollout/num_completed": len(done_indices),
                }, step=self.global_frame)
                self.ep_returns[done_indices] = 0.0
                self.ep_lengths[done_indices] = 0

            self.obs = next_obs
        last_obs = last_obs.copy()
        last_obs_norm = self._normalize_obs(last_obs)
        return RolloutBatch(
            obs=jnp.array(all_obs),
            actions=jnp.array(all_actions),
            rewards=jnp.array(all_rewards),
            terminated=jnp.array(all_terminated),
            truncated=jnp.array(all_truncated),
            log_probs=jnp.array(all_log_probs),
            last_obs=jnp.array(last_obs_norm),
        )

    def train(self):
        cfg = self.cfg
        last_log_frame = 0
        last_eval_frame = 0
        self.obs, _ = self.train_env.reset()
        self.ep_returns = np.zeros(self.num_envs)
        self.ep_lengths = np.zeros(self.num_envs)

        with tqdm(total=cfg.train_frames, desc="training") as pbar:
            while self.global_frame < cfg.train_frames:
                prev_frame = self.global_frame
                rollout_data = self.collect_rollouts()
                self.global_step += self.rollout_length * self.num_envs
                metrics = self.agent.train_step(rollout_data, step=self.global_frame)

                if self.global_frame - last_log_frame >= cfg.log_frames:
                    self.logger.log_scalars("", metrics, step=self.global_frame)
                    last_log_frame = self.global_frame

                if self.global_frame - last_eval_frame >= cfg.eval_frames:
                    self.eval_and_save()
                    last_eval_frame = self.global_frame

                pbar.update(self.global_frame - prev_frame)
            self.eval_and_save()

    def eval_and_save(self):
        returns = np.zeros(self.cfg.eval.num_episodes)
        lengths = np.zeros(self.cfg.eval.num_episodes)

        obs, _ = self.eval_env.reset()
        dones = np.zeros(self.cfg.eval.num_episodes, dtype=bool)

        while not np.all(dones):
            obs_norm = self._normalize_obs(obs)
            actions, _ = self.agent.sample_actions(
                jnp.array(obs_norm), deterministic=True
            )
            actions_clipped = np.clip(actions, -1.0, 1.0)
            obs, rewards, terminated, truncated, infos = self.eval_env.step(actions_clipped)
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


@hydra.main(
    config_path="./config/hb_onpolicy",
    config_name="config",
    version_base=None,
)
def main(cfg: Config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    try:
        _ = cfg.algo.name
    except omegaconf.errors.MissingMandatoryValue:
        err_string = "Algorithm is not specified. Please specify the algorithm via `algo=<algo_name>` in command."
        err_string += "\nAvailable algorithms are:\n  "
        err_string += "\n  ".join(SUPPORTED_AGENTS.keys())
        print(err_string)
        exit(1)

    trainer = HumanoidBenchOnPolicyTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
