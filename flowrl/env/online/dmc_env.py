from collections import deque

import gym
import numpy as np
from dm_control import suite

### define pendulum swingup dense task
from dm_control.suite.pendulum import (
    _ANGLE_BOUND,
    _COSINE_BOUND,
    _DEFAULT_TIME_LIMIT,
    Physics,
    base,
    collections,
    control,
    get_model_and_assets,
    rewards,
)
from dm_control.suite.wrappers import action_scale
from dm_env import specs
from gym.envs.registration import register


class SwingUpDense(base.Task):
  """A Pendulum `Task` to swing up and balance the pole."""

  def __init__(self, random=None):
    """Initialize an instance of `Pendulum`.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super().__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Pole is set to a random angle between [-pi, pi).

    Args:
      physics: An instance of `Physics`.

    """
    physics.named.data.qpos['hinge'] = self.random.uniform(-np.pi, np.pi)
    super().initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation.

    Observations are states concatenating pole orientation and angular velocity
    and pixels from fixed camera.

    Args:
      physics: An instance of `physics`, Pendulum physics.

    Returns:
      A `dict` of observation.
    """
    obs = collections.OrderedDict()
    obs['orientation'] = physics.pole_orientation()
    obs['velocity'] = physics.angular_velocity()
    return obs

  def get_reward(self, physics):
    return rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1), margin=1) # make this task dense reward

def make_pendulum_swingup_dense(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = SwingUpDense(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)
###

class DMControlEnv(object):
    def __init__(
        self,
        task: str,
        seed: int,
        visual: bool,
        frame_skip: int,
        frame_stack: int,
        horizon: int=1000,
        image_size: int = 84,
        camera: int = 0,
    ) -> None:
        super().__init__()

        self.task = task
        self.seed = seed
        self.visual = visual
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.horizon = horizon
        self.image_size = image_size
        self.camera = camera
        self.metadata = {}

        self.domain, self.task = task.split("-")
        if self.domain == "pendulum" and self.task == "swingup_dense":
            self.env = make_pendulum_swingup_dense(random=seed)
            self.env.task.visualize_reward = False
        else:
            self.env = suite.load(self.domain, self.task, task_kwargs={"random": seed}, visualize_reward=False)
        self.env = action_scale.Wrapper(self.env, minimum=-1.0, maximum=1.0)

        if self.visual:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.frame_stack*3, 84, 84), dtype=np.uint8)
        else:
            obs_shape = 0
            for v in self.env.observation_spec().values():
                obs_shape += np.prod(v.shape)
            obs_shape = int(obs_shape)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.env.action_spec().shape),
            high=np.ones(self.env.action_spec().shape),
            dtype=self.env.action_spec().dtype
        )
        self.max_ep_timesteps = (horizon + frame_skip - 1) // frame_skip

        self.queue = deque(maxlen=self.frame_stack)

    def get_obs(self, time_step: object):
        if self.visual: return self.render(self.image_size).transpose(2, 0, 1)
        return np.concatenate([v.flatten() for v in time_step.observation.values()])

    def reset(self):
        self.t = 0
        time_step = self.env.reset()

        obs = self.get_obs(time_step)
        for _ in range(self.frame_stack):
            self.queue.append(obs)
        return np.concatenate(self.queue), {}

    def step(self, action: np.ndarray):
        self.t += 1
        action = action.astype(np.float32)

        reward = 0.0
        for _ in range(self.frame_skip):
            time_step = self.env.step(action)
            reward += time_step.reward

        obs = self.get_obs(time_step)
        self.queue.append(obs)
        return np.concatenate(self.queue), reward, False, self.t == self.max_ep_timesteps, {}

    def render(self, size: int):
        camera = dict(quadruped=2).get(self.domain, self.camera)
        return self.env.physics.render(size, size, camera_id=camera)
