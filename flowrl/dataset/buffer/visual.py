import numpy as np


class VisualReplayBuffer(object):
    def __init__(
        self,
        buffer_size: int,
        nstep: int,
        discount: float,
        frame_stack: int,
        data_specs: dict={}
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.nstep = nstep
        self.discount = discount
        self.frame_stack = frame_stack
        self.full = False

        # tracking and data
        self.index = 0
        self.traj_index = 0
        self.ep_start = True
        self.obs_shape = data_specs["obs"]
        self.action_shape = data_specs["action"]
        self.obs_channels = self.obs_shape[0] // frame_stack
        self.obs = np.zeros([buffer_size, self.obs_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.act = np.zeros([buffer_size, *self.action_shape], dtype=np.float32)
        self.rew = np.zeros([buffer_size, ], dtype=np.float32)
        self.done = np.zeros([buffer_size, ], dtype=np.float32)
        self.valid = np.zeros([buffer_size, ], dtype=np.bool_)

        # constant vectors
        self.discount_vec = np.power(discount, np.arange(nstep)).astype("float32")
        self.next_dis = discount ** nstep

    def add(self, obs, action, next_obs, reward, terminal, timeout):
        if self.ep_start:
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            obs = obs[-self.obs_channels:]
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = obs
                    self.obs[0:end_index] = obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = obs
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.ep_start = False
            self.traj_index += 1
        latest_obs = next_obs[-self.obs_channels:]
        self.obs[self.index] = latest_obs
        self.act[self.index] = action
        self.rew[self.index] = reward
        self.done[self.index] = terminal
        self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
        if self.traj_index >= self.nstep:
            self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
        self.index += 1
        self.traj_index += 1
        if self.index == self.buffer_size:
            self.index = 0
            self.full = True
        if terminal or timeout:
            self.ep_start = True
            self.traj_index = 0

    def sample(self, batch_size):
        indices = np.random.choice(self.valid.nonzero()[0], size=batch_size)
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep) for i in range(batch_size)], axis=0)
        gather_ranges = all_gather_ranges[:, self.frame_stack:]
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]
        sobs_gather_ranges = all_gather_ranges[:, 1:self.frame_stack+1]

        all_rewards = self.rew[gather_ranges]
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)
        obs = np.reshape(self.obs[obs_gather_ranges], [batch_size, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [batch_size, *self.obs_shape])
        sobs = np.reshape(self.obs[sobs_gather_ranges], [batch_size, *self.obs_shape])
        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * (1-self.done[nobs_gather_ranges[:, -1]]), axis=-1)

        return obs, act, rew, dis, nobs, sobs

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index
