
from typing import Optional

import d4rl
import gym
import numpy as np

from flowrl.types import Batch


def compute_returns(reward, end):
    return_ = []
    ep_return = 0
    for i in range(len(reward)):
        ep_return += reward[i]
        if end[i] or i == len(reward) - 1:
            return_.append(ep_return)
            ep_return = 0
    return np.array(return_)

def normalize_reward(reward, end, method="none"):
    if method == "none" or method is None:
        return reward
    elif method == "iql_mujoco":
        returns = compute_returns(reward, end)
        max_return = np.max(returns)
        min_return = np.min(returns)
        factor = 1000.0 / (max_return - min_return)
        print(f"norm reward {method}: return max = {max_return}, return min = {min_return}, factor = {factor}")
        return reward * factor
    elif method == "white":
        r_mean, r_std = np.mean(reward), np.std(reward)
        print(f"norm reward {method}: mean = {r_mean}, std = {r_std}")
        return (reward - r_mean) / r_std
    elif method == "iql_antmaze":
        return reward - 1.0
    elif method == "cql_antmaze":
        return (reward - 0.5) * 4.0
    elif method == "antmaze100":
        return reward * 100.0
    else:
        raise NotImplementedError(f"Reward normalization method {method} not implemented")

def qlearning_dataset(env, dataset=None, terminate_on_end: bool=False, discard_last: bool=True, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            next_actions: An N x dim_action array of next actions.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []
    end_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)  # Thus, the next_obs for the last timestep is totally false
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i+1].astype(np.float32)  # Thus, the next_action for the last timestep is totally false
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        end = False
        episode_step += 1

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps)
        if final_timestep:
            if not done_bool:
                if not terminate_on_end:
                    if discard_last:
                        episode_step = 0
                        end_[-1] = True
                        continue
                else:
                    done_bool = True
        if final_timestep or done_bool:
            end = True
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        end_.append(end)

    end_[-1] = True   # the last traj will be ended whatsoever
    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        "ends": np.array(end_)
    }


class D4RLDataset():
    def __init__(
        self,
        task: str,
        clip_eps: float=0.0,
        scan: bool=True,
        norm_obs: bool=False,
        norm_reward: Optional[str]=None,
    ):
        self.scan = scan
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward

        raw_dataset = qlearning_dataset(gym.make(task))
        dataset = {
            "obs": raw_dataset["observations"],
            "action": raw_dataset["actions"],
            "reward": raw_dataset["rewards"][..., None],
            "terminal": raw_dataset["terminals"][..., None],
            "next_obs": raw_dataset["next_observations"],
            "next_action": raw_dataset["next_actions"],
        }
        if clip_eps > 0:
            dataset["action"] = np.clip(dataset["action"], -(1-clip_eps), 1-clip_eps)
            dataset["next_action"] = np.clip(dataset["next_action"], -(1-clip_eps), 1-clip_eps)
        end = raw_dataset["ends"]

        if self.scan:
            self.scan_index = np.arange(len(dataset["obs"]))
            np.random.shuffle(self.scan_index)
            self.batch_idx = 0

        if norm_obs:
            self.obs_mean = np.mean(dataset["obs"], axis=0)
            self.obs_std = np.std(dataset["obs"], axis=0)
            self.obs_std[self.obs_std == 0] = 1.
            dataset["obs"] = (dataset["obs"] - self.obs_mean[None, ...]) / self.obs_std[None, ...]
            dataset["next_obs"] = (dataset["next_obs"] - self.obs_mean[None, ...]) / self.obs_std[None, ...]
        else:
            self.obs_mean, self.obs_std = 0.0, 1.0

        if norm_reward:
            dataset["reward"] = normalize_reward(dataset["reward"], end, norm_reward)

        self.dataset = dataset
        self.size = len(dataset["obs"])

    def get_obs_stats(self):
        return self.obs_mean, self.obs_std

    def sample(self, batch_size: int):
        if self.scan:
            if self.batch_idx + batch_size >= self.size:
                np.random.shuffle(self.scan_index)
                self.batch_idx = 0
            indices = self.scan_index[self.batch_idx:self.batch_idx + batch_size]
            self.batch_idx += batch_size
        else:
            indices = np.random.randint(0, self.size, batch_size)

        return Batch(
            obs=self.dataset["obs"][indices],
            action=self.dataset["action"][indices],
            reward=self.dataset["reward"][indices],
            terminal=self.dataset["terminal"][indices],
            next_obs=self.dataset["next_obs"][indices],
            next_action=self.dataset["next_action"][indices],
        )
