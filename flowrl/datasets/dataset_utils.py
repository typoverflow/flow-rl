from typing import Callable, Tuple

import d4rl
import gym

import flowrl.wrappers as wrappers
from .d4rl_dataset import D4RLDataset
from .dataset import Dataset
from flowrl.utils.utils import traj_return_normalize


def make_env_and_dataset(env_name: str, seed: int, dataset_name: str,
                         video_save_folder: str = None, reward_tune: str = 'no',
                         episode_return: bool = False, scanning: bool = True, norm_obs=False) -> Tuple[gym.Env, Dataset, Callable]:
    # env = make_env(env_name, seed, video_save_folder)
    env = gym.make(env_name)  # test env. only
    env = wrappers.EpisodeMonitor(env)  # record info['episode']['return', 'length', 'duration']
    env = wrappers.SinglePrecision(env)  # -> np.float32

    if video_save_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_save_folder)

    # set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env, scanning=scanning, norm_obs=norm_obs)

    # reward normalization
    if reward_tune == 'antmaze100':
        def tune_fn(r):
            return r * 100
    elif reward_tune == 'iql_locomotion':
        tune_fn = traj_return_normalize(dataset, scale=1000.)
    elif reward_tune == 'traj_normalize':
        tune_fn = traj_return_normalize(dataset, scale=None)
    elif reward_tune == 'reward_normalize':
        r_mean, r_std = dataset.rewards.mean(), dataset.rewards.std()
        def tune_fn(r):
            return (r - r_mean) / r_std
    elif reward_tune == 'iql_antmaze':
        def tune_fn(r):
            return r - 1.0
    elif reward_tune == 'cql_antmaze':
        def tune_fn(r):
            return (r - 0.5) * 4.0
    elif reward_tune == 'antmaze':
        def tune_fn(r):
            return (r - 0.25) * 2.0
    else:
        tune_fn = None

    # get MC returns?
    if episode_return:
        dataset.episode_returns = env.get_normalized_score(dataset.get_episode_returns()) * 100.

    if tune_fn is not None:
        dataset.rewards = tune_fn(dataset.rewards)

    assert 'd4rl' in dataset_name, "Only support d4rl dataset right now"

    return env, dataset, tune_fn
