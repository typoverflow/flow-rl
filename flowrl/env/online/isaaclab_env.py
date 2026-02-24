from typing import Optional

import gymnasium as gym
import numpy as np


class IsaacLabEnv:
    """Wrapper for IsaacLab environments, adapted from FastTD3.

    Converts all outputs to numpy arrays so the trainer has no torch dependency.
    """

    def __init__(
        self,
        task: str,
        device: str,
        num_envs: int,
        seed: int,
        action_bounds: Optional[float] = None,
    ):
        import torch
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True, device=device)
        simulation_app = app_launcher.app

        import isaaclab_tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(
            task,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.seed = seed
        self.seed = seed
        self._device = device
        self._torch = torch
        self.envs = gym.make(task, cfg=env_cfg, render_mode=None)

        self.num_envs = self.envs.unwrapped.num_envs
        self.max_episode_steps = self.envs.unwrapped.max_episode_length
        self.action_bounds = action_bounds
        self.num_obs = self.envs.unwrapped.single_observation_space["policy"].shape[0]
        self.asymmetric_obs = "critic" in self.envs.unwrapped.single_observation_space
        if self.asymmetric_obs:
            self.num_privileged_obs = self.envs.unwrapped.single_observation_space[
                "critic"
            ].shape[0]
        else:
            self.num_privileged_obs = 0
        self.num_actions = self.envs.unwrapped.single_action_space.shape[0]

    def _to_numpy(self, t) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _to_torch(self, a: np.ndarray):
        return self._torch.from_numpy(np.asarray(a)).float().to(self._device)

    def reset(self, random_start_init: bool = True) -> np.ndarray:
        obs_dict, _ = self.envs.reset()
        # Decorrelate episode horizons (RSL-RL style)
        if random_start_init:
            self.envs.unwrapped.episode_length_buf = self._torch.randint_like(
                self.envs.unwrapped.episode_length_buf, high=int(self.max_episode_steps)
            )
        return self._to_numpy(obs_dict["policy"])

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        actions_torch = self._to_torch(actions)
        if self.action_bounds is not None:
            actions_torch = self._torch.clamp(actions_torch, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = self.envs.step(actions_torch)
        obs = self._to_numpy(obs_dict["policy"])
        return (
            obs,
            self._to_numpy(rew),
            self._to_numpy(terminations.float()),
            self._to_numpy(truncations.float()),
            infos,
        )
