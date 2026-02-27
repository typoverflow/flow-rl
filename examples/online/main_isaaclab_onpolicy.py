import os

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import omegaconf
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from flowrl.agent.online import DPPOAgent, FPOAgent, PPOAgent
from flowrl.config.online.onpolicy_isaaclab_config import Config
from flowrl.dataset.buffer.state import EmpiricalNormalizer
from flowrl.env.online.isaaclab_env import IsaacLabEnv
from flowrl.types import Dict, RolloutBatch
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

jax.config.update("jax_default_matmul_precision", "float32")

SUPPORTED_AGENTS: Dict[str, type] = {
    "ppo": PPOAgent,
    "dppo": DPPOAgent,
    "fpo": FPOAgent,
}


class IsaacLabOnPolicyTrainer:
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

        # Create IsaacLab vectorized environment (returns numpy arrays)
        self.num_envs = cfg.num_envs
        self.rollout_length = cfg.rollout_length
        self.env = IsaacLabEnv(
            task=cfg.task,
            device="cuda:"+str(cfg.device),
            num_envs=cfg.num_envs,
            seed=cfg.seed,
            action_bound=cfg.action_bound,
            disable_bootstrap=cfg.disable_bootstrap,
        )

        self.obs_dim = self.env.num_obs
        self.action_dim = self.env.num_actions
        self.max_episode_steps = self.env.max_episode_steps

        if cfg.norm_obs:
            self.obs_normalizer = EmpiricalNormalizer(shape=(self.obs_dim,))

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
        return self.global_step

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.cfg.norm_obs:
            return self.obs_normalizer.normalize(obs)
        return obs

    def collect_rollouts(self) -> RolloutBatch:
        """Collect rollouts from IsaacLab vectorized env."""
        T = self.rollout_length
        B = self.num_envs

        all_raw_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_actions = np.zeros((T, B, self.action_dim), dtype=np.float32)
        all_next_obs = np.zeros((T, B, self.obs_dim), dtype=np.float32)
        all_rewards = np.zeros((T, B, 1), dtype=np.float32)
        all_terminated = np.zeros((T, B, 1), dtype=np.float32)
        all_truncated = np.zeros((T, B, 1), dtype=np.float32)
        all_extras = None  # Lazily initialized from first sample_actions info

        for t in range(T):
            obs_norm = self._normalize_obs(self.obs)
            all_raw_obs[t] = self.obs.copy()
            all_obs[t] = obs_norm

            actions, info = self.agent.sample_actions(
                jnp.array(obs_norm), deterministic=False
            )
            actions_np = np.array(actions)
            actions_clipped = np.clip(actions_np, -1.0, 1.0)
            all_actions[t] = actions_np

            # Generically collect algorithm-specific info
            if all_extras is None:
                all_extras = {
                    k: np.zeros((T, *np.array(v).shape), dtype=np.float32)
                    for k, v in info.items()
                }
            for k, v in info.items():
                all_extras[k][t] = np.array(v)

            next_obs, rewards, terminated, truncated, infos = self.env.step(actions_clipped)

            all_rewards[t] = rewards[..., np.newaxis]
            all_terminated[t] = terminated[..., np.newaxis]
            all_truncated[t] = truncated[..., np.newaxis]

            next_obs_norm = self._normalize_obs(next_obs)
            all_next_obs[t] = next_obs_norm

            # Track episode stats
            self.ep_returns += rewards
            self.ep_lengths += 1

            done = (terminated + truncated) > 0.5
            done_indices = np.where(done)[0]
            if len(done_indices) > 0:
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

        if self.cfg.norm_obs:
            self.obs_normalizer.update(all_raw_obs.reshape(-1, self.obs_dim))

        return RolloutBatch(
            obs=jnp.array(all_obs),
            actions=jnp.array(all_actions),
            next_obs=jnp.array(all_next_obs),
            rewards=jnp.array(all_rewards),
            terminated=jnp.array(all_terminated),
            truncated=jnp.array(all_truncated),
            extras={k: jnp.array(v) for k, v in all_extras.items()},
        )

    def train(self):
        cfg = self.cfg
        last_log_frame = 0
        last_eval_frame = 0
        self.obs = self.env.reset(random_start_init=True)
        self.ep_returns = np.zeros(self.num_envs)
        self.ep_lengths = np.zeros(self.num_envs)

        self.eval_and_save()
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
        self.logger.close()

    def eval_and_save(self):
        """Evaluate by running the policy for max_episode_steps in the same env."""
        obs = self.env.reset(random_start_init=False)
        eval_returns = np.zeros(self.num_envs)
        eval_lengths = np.zeros(self.num_envs)
        eval_dones = np.zeros(self.num_envs, dtype=bool)

        for _ in range(self.max_episode_steps):
            obs_norm = self._normalize_obs(obs)
            actions, _ = self.agent.sample_actions(
                jnp.array(obs_norm), deterministic=True
            )
            actions_clipped = np.clip(np.array(actions), -1.0, 1.0)
            obs, rewards, terminated, truncated, _ = self.env.step(actions_clipped)

            eval_returns += rewards * (1 - eval_dones)
            eval_lengths += 1 * (1 - eval_dones)
            eval_dones = eval_dones | ((terminated + truncated) > 0.5)

            if np.all(eval_dones):
                break

        eval_metrics = {
            "mean": np.mean(eval_returns),
            "median": np.median(eval_returns),
            "std": np.std(eval_returns),
            "min": np.min(eval_returns),
            "max": np.max(eval_returns),
            "length": np.mean(eval_lengths),
        }
        self.logger.log_scalars("eval", eval_metrics, step=self.global_frame)
        if self.cfg.log.save_ckpt:
            self.agent.save(os.path.join(self.ckpt_save_dir, f"{self.global_frame}"))

        # # Restore training state: reset with decorrelated horizons
        self.obs = self.env.reset(random_start_init=True)
        self.ep_returns = np.zeros(self.num_envs)
        self.ep_lengths = np.zeros(self.num_envs)


@hydra.main(
    config_path="./config/isaaclab_onpolicy",
    config_name="config",
    version_base=None,
)
def main(cfg: Config):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    try:
        _ = cfg.algo.name
    except omegaconf.errors.MissingMandatoryValue:
        err_string = "Algorithm is not specified. Please specify the algorithm via `algo=<algo_name>` in command."
        err_string += "\nAvailable algorithms are:\n  "
        err_string += "\n  ".join(SUPPORTED_AGENTS.keys())
        print(err_string)
        exit(1)

    trainer = IsaacLabOnPolicyTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
