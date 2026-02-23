from __future__ import annotations

from collections import deque

import gymnasium as gym
import humanoid_bench  # noqa: F401 - registers envs
import numpy as np


class HumanoidBenchEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task: str,
        seed: int,
        frame_skip: int,
        frame_stack: int,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self._seed = seed
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.render_mode = render_mode

        self.env = gym.make(task, autoreset=False, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.queue: deque = deque(maxlen=self.frame_stack)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        seed = seed if seed is not None else self._seed
        obs, info = self.env.reset(seed=seed, options=options or {})
        self.queue.clear()
        for _ in range(self.frame_stack):
            self.queue.append(obs)
        return np.concatenate(self.queue), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        self.queue.append(obs)
        return np.concatenate(self.queue), total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
