import d4rl
import gym
import numpy as np

from flowrl.env.gym_wrappers import EpisodeMonitor


def make_env(env_name, seed):
    """Make D4RL environment."""
    env = gym.make(env_name)
    env.seed(seed)
    env = EpisodeMonitor(env)
    return env
