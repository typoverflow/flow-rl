from __future__ import annotations

import gymnasium as gym
import humanoid_bench

from gymnasium.wrappers import TimeLimit
import numpy as np


def make_env(env_name, rank, render_mode=None, seed=0):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """

    if env_name in [
        "h1hand-push-v0",
        "h1-push-v0",
        "h1hand-cube-v0",
        "h1cube-v0",
        "h1hand-basketball-v0",
        "h1-basketball-v0",
        "h1hand-kitchen-v0",
        "h1-kitchen-v0",
    ]:
        max_episode_steps = 500
    else:
        max_episode_steps = 1000

    def _init():
        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)

        return env, max_episode_steps

    return _init


class HumanoidBenchEnv:
    """Wraps HumanoidBench environment to support parallel environments."""

    def __init__(self, env_name, num_envs=1, render_mode=None): #, device=None
        # NOTE: HumanoidBench action space is already normalized to [-1, 1]
        self.num_envs = num_envs

        # Create the base environment
        self.env, self.max_episode_steps = make_env(env_name, 0, render_mode=render_mode)()
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        # For compatibility with MuJoCo Playground
        self.asymmetric_obs = False  # For compatibility with MuJoCo Playground
        self.num_obs = self.observation_space.shape[-1]
        self.num_actions = self.action_space.shape[-1]

    def reset(self):
        """Reset the environment."""
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def render(self):
        assert (
            self.num_envs == 1
        ), "Currently only supports single environment rendering"
        return self.env.render()

    def step(self, actions):
        actions = actions.astype(np.float32)

        obs, reward, terminated, truncated, raw_info = self.env.step(actions)

        return obs, reward, terminated, truncated, raw_info