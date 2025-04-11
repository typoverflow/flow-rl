
from typing import Optional

import d4rl
import gym
import numpy as np
import d4rl
from collections import namedtuple

from .base import Batch

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
        
        raw_dataset = d4rl.qlearning_dataset(gym.make(task))
        dataset = {
            "obs": raw_dataset["observations"],
            "action": raw_dataset["actions"],
            "reward": raw_dataset["rewards"],
            "terminal": raw_dataset["terminals"],
            "next_obs": raw_dataset["next_observations"],
        }
        if clip_eps > 0:
            dataset["action"] = np.clip(dataset["action"], -(1-clip_eps), 1-clip_eps)

        end = np.zeros_like(dataset["reward"])
        for i in range(len(end) - 1):
            if dataset["terminal"][i] or \
                np.linalg.norm(dataset["obs"][i+1]-dataset["next_obs"][i]) > 1e-6:
                end[i] = 1
            else:
                end[i] = 0
        end[-1] = 1

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
        )

