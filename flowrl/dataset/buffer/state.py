from typing import Any

import numpy as np

from flowrl.types import Batch


class RMSNormalizer():
    def __init__(self, epsilon=1e-8, shape=(), dtype=np.float32):
        super().__init__()
        self.mean = np.zeros(shape, dtype=dtype)
        self.mean_square = np.ones(shape, dtype=dtype)
        self.count = epsilon
        self.epsilon = 1e-8

    def update(self, x):
        x = np.asarray(x, dtype=self.mean.dtype)
        if x.ndim == 1:
            batch_mean = x
            batch_mean_sq = x**2
            n = 1
        else:
            batch_mean = np.mean(x, axis=0)
            batch_mean_sq = np.mean(x**2, axis=0)
            n = x.shape[0]
        self.count += n
        self.mean = self.mean + (batch_mean - self.mean) * n / self.count
        self.mean_square = self.mean_square + (batch_mean_sq - self.mean_square) * n / self.count

    def normalize(self, x):
        var = np.clip(self.mean_square - self.mean**2, a_min=0, a_max=None)
        std = np.sqrt(var + self.epsilon)
        std = np.clip(std, a_min=1e-4, a_max=None)
        return (x - self.mean) / std


class EmpiricalNormalizer():
    def __init__(self, epsilon=1e-8, shape=(), dtype=np.float32):
        super().__init__()
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = 1e-8

    def update(self, x):
        global_batch_size = x.shape[0]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        new_count = self.count + global_batch_size

        delta = batch_mean - self.mean
        self.mean += delta * global_batch_size / new_count

        m_a = self.var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta**2 * (self.count * global_batch_size / new_count)
        self.var = M2 / new_count
        self.std = np.sqrt(self.var)
        self.count = new_count

    def normalize(self, x):
        return (x - self.mean) / np.clip(self.std, a_min=1e-4, a_max=None)


class ReplayBuffer(object):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_size: int=int(1e6),
        lap_alpha: float=0.0,
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        if lap_alpha > 0:
            from flowrl.data_structure import MinTree, SumTree
            self.lap = True
            self.sum_tree = SumTree(max_size)
            self.min_tree = MinTree(max_size)
            self.metric_fn = lambda x: np.power(np.clip(x, a_min=1.0, a_max=None), lap_alpha)
            self.max_priority = 1.0
        else:
            self.lap = False

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        if self.lap:
            self.sum_tree.add(self.max_priority)
            self.min_tree.add(-self.max_priority)

    def sample(self, batch_size):
        if self.lap:
            target = np.random.random(size=(batch_size, ))
            indices = self.sum_tree.find(target, scale=True)[0]
            indices = np.asarray(indices)
        else:
            indices = np.random.choice(self.size, size=batch_size, replace=True)

        return Batch(
            obs=self.state[indices],
            action=self.action[indices],
            reward=self.reward[indices],
            terminal=self.done[indices],
            next_obs=self.next_state[indices],
            next_action=None,
        ), indices

    def update(self, indices, priorities):
        if self.lap:
            priorities = self.metric_fn(priorities)
            self.max_priority = max(self.max_priority, priorities.max())
            self.sum_tree.update(indices, priorities)
            self.min_tree.update(indices, -priorities)
        else:
            raise ValueError("Only LAP buffer can be updated")

    def reset_max_priority(self):
        if self.lap:
            self.max_priority = - self.min_tree.min()
        else:
            raise ValueError("Only LAP buffer can reset max priority")
