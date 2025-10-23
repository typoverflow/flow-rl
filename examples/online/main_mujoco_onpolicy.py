import os
from typing import Dict, Tuple

import gymnasium as gym
import gymnasium_robotics
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from flowrl.agent.online import *
from flowrl.config.online.mujoco import Config
from flowrl.types import *
from flowrl.utils.logger import CompositeLogger
from flowrl.utils.misc import set_seed_everywhere

jax.config.update("jax_default_matmul_precision", "float32")

SUPPORTED_AGENTS: Dict[str, BaseAgent] = {
    "ppo": PPOAgent,
}


class OnPolicyTrainer():
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

        # Environment setup
        self.env = gym.make(cfg.task)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
        self.env = gym.wrappers.ClipAction(self.env)

        # Agent setup
        agent_cls = SUPPORTED_AGENTS[cfg.algo.name.lower()]
        self.agent = agent_cls(
            obs_dim=self.env.observation_space.shape[0],
            act_dim=self.env.action_space.shape[0],
            cfg=cfg.algo,
            seed=cfg.seed,
        )

        # Storage for rollouts
        self.rollout_buffer = {
            'obs': np.zeros((cfg.rollout_steps, self.env.observation_space.shape[0]), dtype=np.float32),
            'actions': np.zeros((cfg.rollout_steps, self.env.action_space.shape[0]), dtype=np.float32),
            'logprobs': np.zeros((cfg.rollout_steps,), dtype=np.float32),
            'rewards': np.zeros((cfg.rollout_steps,), dtype=np.float32),
            'dones': np.zeros((cfg.rollout_steps,), dtype=np.float32),
            'values': np.zeros((cfg.rollout_steps,), dtype=np.float32),
        }

        # Training statistics
        self.global_step = 0
        self.episode_returns = []
        self.episode_lengths = []

    def collect_rollout(self, obs: np.ndarray) -> np.ndarray:
        """Collect a rollout of experiences."""

        for step in range(self.cfg.rollout_steps):
            # Convert to JAX arrays
            obs_jax = jnp.array(obs[None, :])

            # Sample action
            action, info = self.agent.sample_actions(obs_jax, deterministic=False)
            logprob = info['logprob']

            # Get value estimate
            value = self.agent.critic(obs_jax, training=False)

            # Step environment
            next_obs, reward, terminated, truncated, info_env = self.env.step(np.array(action[0]))
            done = terminated or truncated

            # Store experience
            self.rollout_buffer['obs'][step] = obs
            self.rollout_buffer['actions'][step] = action[0]
            self.rollout_buffer['logprobs'][step] = logprob[0]
            self.rollout_buffer['rewards'][step] = reward
            self.rollout_buffer['dones'][step] = done
            self.rollout_buffer['values'][step] = value[0, 0]

            obs = next_obs
            self.global_step += 1

            # Log episode statistics
            if done:
                if 'episode' in info_env:
                    self.episode_returns.append(info_env['episode']['r'])
                    self.episode_lengths.append(info_env['episode']['l'])

                    if len(self.episode_returns) % 10 == 0:
                        print(f"Step: {self.global_step}, Return: {info_env['episode']['r']:.2f}, Length: {info_env['episode']['l']}")

                        if len(self.episode_returns) > 0:
                            self.logger.log(
                                {
                                    "charts/episodic_return": np.mean(self.episode_returns[-100:]),
                                    "charts/episodic_length": np.mean(self.episode_lengths[-100:]),
                                    "charts/SPS": int(self.global_step / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0),
                                },
                                step=self.global_step,
                            )

                obs, _ = self.env.reset()

        return obs

    def compute_gae_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        obs = self.rollout_buffer['obs']
        rewards = self.rollout_buffer['rewards']
        dones = self.rollout_buffer['dones']
        values = self.rollout_buffer['values']

        # Get bootstrap value for final observation
        final_obs = jnp.array(obs[-1:])
        next_value = self.agent.critic(final_obs, training=False)[0, 0]

        # Convert to JAX arrays
        rewards_jax = jnp.array(rewards)
        dones_jax = jnp.array(dones)
        values_jax = jnp.array(values)

        # Compute GAE advantages
        advantages = jnp.zeros_like(rewards_jax)
        last_gae_lam = 0

        for t in reversed(range(self.cfg.rollout_steps)):
            if t == self.cfg.rollout_steps - 1:
                next_non_terminal = 1.0 - dones_jax[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones_jax[t + 1]
                next_values = values_jax[t + 1]

            delta = rewards_jax[t] + self.cfg.gamma * next_values * next_non_terminal - values_jax[t]
            advantages = advantages.at[t].set(
                delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * last_gae_lam
            )
            last_gae_lam = advantages[t]

        # Compute returns
        returns = advantages + values_jax

        return np.array(advantages), np.array(returns)

    def train_iteration(self, obs: np.ndarray) -> np.ndarray:
        """Perform one training iteration (collect rollout + update policy)."""

        # Collect rollout
        obs = self.collect_rollout(obs)

        # Compute advantages and returns
        advantages, returns = self.compute_gae_advantages()

        # Prepare batch for training
        batch_data = {
            'obs': self.rollout_buffer['obs'],
            'actions': self.rollout_buffer['actions'],
            'logprobs': self.rollout_buffer['logprobs'],
            'advantages': advantages,
            'return_to_go': returns,
            'values': self.rollout_buffer['values'],
        }

        # Normalize advantages if configured
        if self.cfg.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            batch_data['advantages'] = advantages

        # Convert to JAX arrays
        batch = Batch(
            obs=jnp.array(batch_data['obs']),
            action=jnp.array(batch_data['actions']),
            logprob=jnp.array(batch_data['logprobs']),
            advantage=jnp.array(batch_data['advantages']),
            return_to_go=jnp.array(batch_data['return_to_go']),
            value=jnp.array(batch_data['values']),
            # Dummy fields (not used in PPO)
            reward=jnp.zeros(self.cfg.rollout_steps),
            next_obs=jnp.zeros_like(jnp.array(batch_data['obs'])),
            terminal=jnp.zeros(self.cfg.rollout_steps),
        )

        # Perform multiple epochs of updates
        for epoch in range(self.cfg.update_epochs):
            metrics = self.agent.train_step(batch, step=self.global_step)

            # Log metrics
            if epoch == 0:  # Only log first epoch metrics to avoid clutter
                metrics_dict = {f"losses/{k}": v for k, v in metrics.items()}
                self.logger.log(metrics_dict, step=self.global_step)

        return obs

    def train(self):
        """Main training loop."""
        self.start_time = time.time()

        # Initialize environment
        obs, _ = self.env.reset()

        # Training loop
        for iteration in tqdm(range(self.cfg.num_iterations), desc="Training"):
            obs = self.train_iteration(obs)

            # Save checkpoint periodically
            if (iteration + 1) % self.cfg.save_freq == 0:
                ckpt_path = os.path.join(self.ckpt_save_dir, f"step_{self.global_step}")
                os.makedirs(self.ckpt_save_dir, exist_ok=True)
                self.agent.save(ckpt_path)
                print(f"Saved checkpoint at step {self.global_step}")

        # Final checkpoint
        ckpt_path = os.path.join(self.ckpt_save_dir, "final")
        self.agent.save(ckpt_path)
        print(f"Training completed. Final checkpoint saved at step {self.global_step}")

        self.env.close()
        self.logger.close()


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: Config) -> None:
    trainer = OnPolicyTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    import time
    main()
