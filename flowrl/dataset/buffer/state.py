import numpy as np

from flowrl.types import Batch


class RunningMeanStdNormalizer():
    def __init__(self, epsilon=1e-8, shape=(), dtype=np.float32):
        super().__init__()
        self.mean = np.zeros(shape, dtype=dtype)
        self.mean_square = np.zeros(shape, dtype=dtype)
        self.count = epsilon
        self.epsilon = 1e-8

    def update(self, x):
        self.count += 1
        delta1 = np.squeeze(x) - self.mean
        self.mean += delta1 / self.count
        delta2 = np.squeeze(x)**2 - self.mean_square
        self.mean_square += delta2 / self.count

    def normalize(self, x):
        var = np.clip(self.mean_square - self.mean**2, a_min=0, a_max=None)
        std = np.sqrt(var + self.epsilon)
        std = np.clip(std, a_min=1e-4, a_max=None)
        return (x - self.mean) / std


class ReplayBuffer(object):
    def __init__(self, obs_dim, action_dim, max_size=int(1e6), norm_obs=False, norm_reward=False):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        if norm_obs:
            self.obs_rms = RunningMeanStdNormalizer(shape=[obs_dim, ])
        if norm_reward:
            self.reward_rms = RunningMeanStdNormalizer(shape=[1, ])

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.norm_obs:
            self.obs_rms.update(state)
        if self.norm_reward:
            self.reward_rms.update(reward)

    def sample(self, batch_size):
        ind = np.random.choice(self.size, size=batch_size, replace=False)
        return Batch(
            obs=self.state[ind] if not self.norm_obs else self.obs_rms.normalize(self.state[ind]),
            action=self.action[ind],
            reward=self.reward[ind] if not self.norm_reward else self.reward_rms.normalize(self.reward[ind]),
            terminal=self.done[ind],
            next_obs=self.next_state[ind] if not self.norm_obs else self.obs_rms.normalize(self.next_state[ind]),
            next_action=None,
        )

    def normalize_obs(self, obs):
        if self.norm_obs:
            return self.obs_rms.normalize(obs)
        return obs
