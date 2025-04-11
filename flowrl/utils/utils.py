from typing import Optional

import gym
import numpy as np
from gym.wrappers import RescaleAction
from gym.wrappers.pixel_observation import PixelObservationWrapper
from tqdm import tqdm

import flowrl.wrappers as wrappers


def get_episodic_data():
    import collections
    import pickle

    import d4rl
    import gym
    for env_name in ['halfcheetah', 'hopper', 'walker2d']:
        for dataset_type in ['medium', 'medium-replay', 'expert']:
            name = f'{env_name}-{dataset_type}-v2'
            env = gym.make(name)
            dataset = d4rl.qlearning_dataset(env)  # env.get_dataset()

            N = dataset['rewards'].shape[0]
            data_ = collections.defaultdict(list)

            use_timeouts = False
            if 'timeouts' in dataset:
                use_timeouts = True

            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset['terminals'][i])
                if use_timeouts:
                    final_timestep = dataset['timeouts'][i]
                else:
                    final_timestep = (episode_step == 1000 - 1)
                for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
                    data_[k].append(dataset[k][i])
                if done_bool or final_timestep:
                    episode_step = 0
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            returns = np.array([np.sum(p['rewards']) for p in paths])
            num_samples = np.sum([p['rewards'].shape[0] for p in paths])
            print(f'Number of samples collected: {num_samples}')
            print(
                f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

            with open(f'datasets/{name}.pkl', 'wb') as f:
                pickle.dump(paths, f)


def make_env(env_name: str,
             seed: int,
             save_folder: Optional[str] = None,
             add_episode_monitor: bool = True,
             action_repeat: int = 1,
             frame_stack: int = 1,
             from_pixels: bool = False,
             pixels_only: bool = True,
             image_size: int = 84,
             sticky: bool = False,
             gray_scale: bool = False,
             flatten: bool = True,
             obs_mean: float = None,
             obs_std: float = None) -> gym.Env:
    # Check if the env is in gym.
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        env = gym.make(env_name)
    else:
        domain_name, task_name = env_name.split('-')
        env = wrappers.DMCEnv(domain_name=domain_name,
                              task_name=task_name,
                              task_kwargs={'random': seed})

    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if add_episode_monitor:
        env = wrappers.EpisodeMonitor(env)

    if action_repeat > 1:
        env = wrappers.RepeatAction(env, action_repeat)

    env = RescaleAction(env, -1.0, 1.0)

    if save_folder is not None:
        env = gym.wrappers.RecordVideo(env, save_folder)

    if from_pixels:
        if env_name in env_ids:
            camera_id = 0
        else:
            camera_id = 2 if domain_name == 'quadruped' else 0
        env = PixelObservationWrapper(env,
                                      pixels_only=pixels_only,
                                      render_kwargs={
                                          'pixels': {
                                              'height': image_size,
                                              'width': image_size,
                                              'camera_id': camera_id
                                          }
                                      })
        env = wrappers.TakeKey(env, take_key='pixels')
        if gray_scale:
            env = wrappers.RGB2Gray(env)
    else:
        env = wrappers.SinglePrecision(env)

    if frame_stack > 1:
        env = wrappers.FrameStack(env, num_stack=frame_stack)

    if sticky:
        env = wrappers.StickyActionEnv(env)

    if obs_mean is not None and obs_std is not None:
        env = wrappers.TransformObservation(env, lambda obs: (obs-obs_mean)/obs_std)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env


def sample_n_k(n, k):
    """Sample k distinct elements uniformly from range(n)"""
    """it is faster to get replace=False"""

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return np.empty((0,), dtype=np.int64)
    elif 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        result = np.random.choice(n, 2 * k)
        selected = set()
        selected_add = selected.add
        j = k
        for i in range(k):
            x = result[i]
            while x in selected:
                x = result[i] = result[j]
                j += 1
                if j == 2 * k:
                    # This is slow, but it rarely happens.
                    result[k:] = np.random.choice(n, k)
                    j = k
            selected_add(x)
        return result[:k]


def prepare_output_dir(
        folder='results',
        time_stamp=True,
        time_format="%Y%m%d-%H%M%S",
        suffix: str = "") -> str:
    import datetime
    import os
    """Prepare a directory for outputting training results.
    Returns:
        Path of the output directory created by this function (str).
    """

    if time_stamp:
        suffix = str(datetime.datetime.now().strftime(time_format)) + "_" + suffix

    # basedir -> 'results'
    folder = os.path.join(folder or ".", suffix)
    # assert not os.path.exists(out_dir), "found existed experiment folder!"

    os.makedirs(folder, exist_ok=True)

    # Save all the environment variables
    # with open(os.path.join(out_dir, "environ.txt"), "w") as f:
    #     f.write(json.dumps(dict(os.environ)))
    return folder


# @ray.remote
# class StepMonitor:
#     def __init__(self, n_bars: int = 1):
#         self.passed_steps = [0] * n_bars

#     def update(self, i):
#         self.passed_steps[i] += 1

#     def get_steps(self):
#         return self.passed_steps


# class MBars:

#     def __init__(self, total: int, title: str, n_bars: int = 1):
#         self.max_steps = total
#         self.bars = [tqdm(total=total,
#                           desc=f'Train {title}[{i}]',
#                           smoothing=0.01,
#                           colour='GREEN',
#                           position=i) for i in range(n_bars)]
#         self.process = StepMonitor.remote(n_bars)

#     def flush(self):
#         import time
#         while True:
#             for step, bar in zip(ray.get(self.process.get_steps.remote()), self.bars):
#                 bar.update(step - bar.n)
#             time.sleep(0.1)

#             if all(bar.n >= self.max_steps for bar in self.bars):
#                 print('All process finished!')
#                 return


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]
    returns = []
    episode_return = 0

    for i in tqdm(range(len(observations)), desc='split to trajectories'):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        episode_return += rewards[i]
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])
            returns.append(episode_return)
            episode_return = 0
    return trajs, returns


def traj_return_normalize(dataset, scale=None):
    """
    iql_normalize: normalized reward <- reward /(max_return-min_return)* 1000.0
    seed https://github.com/ikostrikov/implicit_q_learning/blob/master/train_offline.py
    """
    trajs, returns = split_into_trajectories(dataset.observations, dataset.actions,
                                             dataset.rewards, dataset.masks,
                                             dataset.dones_float,
                                             dataset.next_observations)

    if scale is None:  # using the average trajectory length
        scale = int(np.mean([len(_) for _ in trajs]))
    assert scale > 0

    def compute_returns(traj):
        episode_return = 0
        for _, _, r, _, _, _ in traj:
            episode_return += r
        return episode_return

    trajs.sort(key=compute_returns)

    # return dataset.rewards / (compute_returns(trajs[-1]) - compute_returns(trajs[0])) * scale
    # dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    # dataset.rewards = tune_fn(dataset.rewards)

    # if negative:
    #     dataset.rewards -= max(dataset.rewards)
    # dataset.rewards *= scale

    traj_gap = compute_returns(trajs[-1]) - compute_returns(trajs[0])

    def tune_fn(r):
        return scale * r / traj_gap

    return tune_fn

# def iql_normalize(reward, not_done):
#     trajs_rt = []
#     episode_return = 0.0
#     for i in range(len(reward)):
#         episode_return += reward[i]
#         if not not_done[i]:
#             trajs_rt.append(episode_return)
#             episode_return = 0.0
#     rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
#     reward /= (rt_max - rt_min)
#     reward *= 1000.
#     return reward
