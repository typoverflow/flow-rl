import d4rl
import gymnasium
import numpy as np

from flowrl.env.gym_wrappers import EpisodeMonitor


def make_env(env_name, seed):
    """Make D4RL environment."""
    env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    env = EpisodeMonitor(env)
    return env
