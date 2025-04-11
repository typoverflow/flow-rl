
import d4rl
import gym
import numpy as np

from .dataset import Dataset


class D4RLDataset(Dataset):

    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 scanning: bool = True,
                 norm_obs: bool = False
                 ):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            """
            Here the dones signal is given if the observation does not change.
            Dones signal is used to separate trajectories
            """
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        if norm_obs:
            self.obs_mean = np.mean(dataset["observations"], axis=0)
            self.obs_std = np.std(dataset["observations"], axis=0)
            self.obs_std[self.obs_std==0] = 1.
            dataset["observations"] = (dataset["observations"] - self.obs_mean[None, ...]) / self.obs_std[None, ...]
            dataset["next_observations"] = (dataset["next_observations"] - self.obs_mean[None, ...]) / self.obs_std[None, ...]
        else:
            self.obs_mean, self.obs_std = 0., 1.

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']),
                         scanning=scanning)
