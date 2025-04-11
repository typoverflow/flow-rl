
from typing import Tuple, Union

import numpy as np
from tqdm import tqdm

from flowrl.networks.types import Batch
from flowrl.utils.utils import sample_n_k


def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _, _ in traj:
        episode_return += rew

    return episode_return


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


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
        next_observations)


class Dataset(object):
    """
    mask = 1 - terminal, which is used for Q <- r + mask*(discount*Q_next)
    """

    def __init__(self,
                 observations: np.ndarray,
                 actions: np.ndarray,
                 rewards: np.ndarray,
                 masks: np.ndarray,
                 dones_float: np.ndarray,
                 next_observations: np.ndarray,
                 size: int,
                 scanning: bool = False,  # scanning=True for offlineRL, may not be suitable for online replay buffer
                 ):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

        self.scanning = scanning
        if self.scanning:
            self.scanning_indices = np.arange(self.size)
            np.random.shuffle(self.scanning_indices)
            self.batch_idx = 0

    def sample(self, batch_size: int) -> Batch:

        if self.scanning:
            if self.batch_idx + batch_size >= self.size:
                np.random.shuffle(self.scanning_indices)
                self.batch_idx = 0
            indices = self.scanning_indices[self.batch_idx:self.batch_idx + batch_size]
            self.batch_idx += batch_size
        else:
            indices = sample_n_k(self.size, batch_size)

        return Batch(observations=self.observations[indices],
                     actions=self.actions[indices],
                     rewards=self.rewards[indices],
                     masks=self.masks[indices],
                     next_observations=self.next_observations[indices],)
                     # mc_return=self.mc_return[indices])

    def get_initial_states(
            self,
            and_action: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        states = []
        if and_action:
            actions = []
        trajs, _ = split_into_trajectories(self.observations, self.actions,
                                           self.rewards, self.masks,
                                           self.dones_float,
                                           self.next_observations)

        trajs.sort(key=compute_returns)

        for traj in trajs:
            states.append(traj[0][0])
            if and_action:
                actions.append(traj[0][1])

        states = np.stack(states, 0)
        if and_action:
            actions = np.stack(actions, 0)
            return states, actions
        else:
            return states

    def get_future_mc_return(self, discount: float, avg_credit_assign: bool = False) -> np.ndarray:
        """
        more useful for training Q-functions
        """
        trajs, _ = split_into_trajectories(self.observations, self.actions,
                                           self.rewards, self.masks,
                                           self.dones_float,
                                           self.next_observations)
        mc_returns = []
        for traj in trajs:
            mc_return = 0.0
            for i, (_, _, reward, _, _, _) in enumerate(traj[::-1]):
                mc_return = reward + discount * mc_return
                scale = i + 1 if avg_credit_assign else 1  # avg. by the rest trajectory length
                mc_returns.append(mc_return / scale)
        return np.asarray(mc_returns[::-1])

    def get_traj_wise_mc_return(self, discount: float, avg_credit_assign: bool) -> np.ndarray:
        """
        set the trajectory mc-return to each transition component
        """
        trajs, _ = split_into_trajectories(self.observations, self.actions,
                                           self.rewards, self.masks,
                                           self.dones_float,
                                           self.next_observations)
        mc_returns = []
        for traj in trajs:
            mc_return = 0.0
            for i, (_, _, reward, _, _, _) in enumerate(traj):
                mc_return += reward * (discount ** i)
            if avg_credit_assign:
                mc_return /= len(traj)  # avg. credit assignment
            mc_returns += [mc_return] * len(traj)  # set to each transition
        return np.asarray(mc_returns)

    def get_episode_returns(self) -> np.ndarray:
        _, returns = split_into_trajectories(self.observations, self.actions,
                                             self.rewards, self.masks,
                                             self.dones_float,
                                             self.next_observations)

        return np.asarray(returns)
        # mc_returns = []
        # for traj in trajs:
        #     mc_return = 0.0
        #     for i, (_, _, reward, _, _, _) in enumerate(traj):
        #         mc_return += reward * (discount ** i)
        #     mc_returns.append(mc_return)
        #
        # return np.asarray(mc_returns)

    # def normalize_mc_return(self, num_bins: int = 0, shift: float = 1):
    #     """
    #     normalize to [0, 1], if num_bins is given, give the bins indices from 1, 2, ..., N
    #     """
    #     min_ret, max_ret = min(self.mc_return), max(self.mc_return)
    #     normed_ret = (self.mc_return - min_ret) / (max_ret - min_ret + 1e-5)
    #
    #     if num_bins > 0:  # discrete class label
    #         normed_ret = np.floor(normed_ret * num_bins).astype(int) + 1  # [1, 2, ..., N]
    #     else:  # otherwise continuous in [1, 2], make ret=0 as un-conditional label
    #         normed_ret += shift
    #
    #     return normed_ret

    def take_top(self, percentile: float = 100.0):
        assert percentile > 0.0 and percentile <= 100.0

        trajs, _ = split_into_trajectories(self.observations, self.actions,
                                           self.rewards, self.masks,
                                           self.dones_float,
                                           self.next_observations)

        trajs.sort(key=compute_returns)

        N = int(len(trajs) * percentile / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def take_random(self, percentage: float = 100.0):
        assert percentage > 0.0 and percentage <= 100.0

        trajs, _ = split_into_trajectories(self.observations, self.actions,
                                           self.rewards, self.masks,
                                           self.dones_float,
                                           self.next_observations)
        np.random.shuffle(trajs)

        N = int(len(trajs) * percentage / 100)
        N = max(1, N)

        trajs = trajs[-N:]

        (self.observations, self.actions, self.rewards, self.masks,
         self.dones_float, self.next_observations) = merge_trajectories(trajs)

        self.size = len(self.observations)

    def train_validation_split(self,
                               train_fraction: float = 0.8
                               ) -> Tuple['Dataset', 'Dataset']:
        trajs, _ = split_into_trajectories(self.observations, self.actions,
                                           self.rewards, self.masks,
                                           self.dones_float,
                                           self.next_observations)
        train_size = int(train_fraction * len(trajs))

        np.random.shuffle(trajs)

        (train_observations, train_actions, train_rewards, train_masks,
         train_dones_float,
         train_next_observations) = merge_trajectories(trajs[:train_size])

        (valid_observations, valid_actions, valid_rewards, valid_masks,
         valid_dones_float,
         valid_next_observations) = merge_trajectories(trajs[train_size:])

        train_dataset = Dataset(train_observations,
                                train_actions,
                                train_rewards,
                                train_masks,
                                train_dones_float,
                                train_next_observations,
                                size=len(train_observations))
        valid_dataset = Dataset(valid_observations,
                                valid_actions,
                                valid_rewards,
                                valid_masks,
                                valid_dones_float,
                                valid_next_observations,
                                size=len(valid_observations))

        return train_dataset, valid_dataset
