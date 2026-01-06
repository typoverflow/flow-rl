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
    ) -> None:
        super().__init__()
        self.task = task
        self.seed = seed
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack

        self.env = gym.make(task, autoreset=False)
        self.env.seed(seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.queue = deque(maxlen=self.frame_stack)

    def reset(self):
        obs, info = self.env.reset()
        for _ in range(self.frame_stack):
            self.queue.append(obs)
        return np.concatenate(self.queue), info

    def step(self, action):
        action = action.astype(np.float32)
        reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward += reward
            if terminated or truncated:
                break
        self.queue.append(obs)
        return np.concatenate(self.queue), reward, terminated, truncated, info
