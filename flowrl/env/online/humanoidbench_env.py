from __future__ import annotations

from collections import deque

import gymnasium as gym
import humanoid_bench
import numpy as np
from humanoid_bench.env import ROBOTS, TASKS


class HumanoidBenchEnv:
    def __init__(
        self,
        task: str,
        seed: int,
        frame_skip: int,
        frame_stack: int,
        horizon: int = 1000,
    ) -> None:
        super().__init__()
        self.task = task
        self.seed = seed
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.horizon = horizon

        self.env = gym.make(task, max_episode_steps=horizon, autoreset=False)
        self.env.seed(seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.max_ep_timesteps = (horizon + frame_skip - 1) // frame_skip
        self.queue = deque(maxlen=self.frame_stack)

    def reset(self):
        self.t = 0
        obs, info = self.env.reset()
        for _ in range(self.frame_stack):
            self.queue.append(obs)
        return np.concatenate(self.queue), info

    def step(self, action):
        self.t += 1
        action = action.astype(np.float32)
        reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward += reward
            if terminated or truncated:
                break
        self.queue.append(obs)
        return np.concatenate(self.queue), reward, terminated, truncated, info
