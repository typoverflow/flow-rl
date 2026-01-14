from __future__ import annotations

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv


def _make_env(task: str, seed: int, frame_skip: int, frame_stack: int):
    def _init():
        import warnings
        warnings.filterwarnings("ignore", message=".*DISPLAY environment variable.*")

        import gymnasium as gym
        import humanoid_bench
        from collections import deque

        env = gym.make(task, render_mode=None, autoreset=False)
        env.unwrapped.seed(seed)

        class _Wrapper(gym.Wrapper):
            def __init__(self, env, frame_skip, frame_stack):
                super().__init__(env)
                self.frame_skip = frame_skip
                self.frame_stack = frame_stack
                self.queue = deque(maxlen=frame_stack)
                obs_shape = env.observation_space.shape
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(obs_shape[0] * frame_stack,),
                    dtype=np.float32
                )

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                for _ in range(self.frame_stack):
                    self.queue.append(obs)
                return np.concatenate(self.queue).astype(np.float32), info

            def step(self, action):
                action = action.astype(np.float32)
                total_reward = 0.0
                for _ in range(self.frame_skip):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    if terminated or truncated:
                        break
                self.queue.append(obs)
                return np.concatenate(self.queue).astype(np.float32), total_reward, terminated, truncated, info

        return _Wrapper(env, frame_skip, frame_stack)
    return _init


class HumanoidBenchEnv:
    def __init__(self, task: str, num_envs: int, seed: int, frame_skip: int, frame_stack: int):
        self.num_envs = num_envs
        self.envs = SubprocVecEnv(
            [_make_env(task, seed + i, frame_skip, frame_stack) for i in range(num_envs)],
            start_method='spawn'
        )
        self.observation_space = self.envs.observation_space
        self.action_space = self.envs.action_space

    def reset(self):
        return self.envs.reset(), {}

    def step(self, actions):
        obs, rewards, dones, infos = self.envs.step(actions)
        terminated = np.array([info.get("TimeLimit.truncated", False) == False and dones[i]
                               for i, info in enumerate(infos)])
        truncated = np.array([info.get("TimeLimit.truncated", False) for info in infos])
        final_obs = np.copy(obs)
        for i in range(self.num_envs):
            if dones[i] and "terminal_observation" in infos[i]:
                final_obs[i] = infos[i]["terminal_observation"]
        return obs, rewards, terminated, truncated, {"final_obs": final_obs}

    def close(self):
        self.envs.close()
