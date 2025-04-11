import d4rl
import gymnasium
import numpy as np

from flowrl.envs.gym_wrappers import EpisodeMonitor


def make_env(env_name):
    """Make D4RL environment."""
    env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env = EpisodeMonitor(env)
    return env


