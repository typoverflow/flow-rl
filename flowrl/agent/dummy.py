from flowrl.config.d4rl.algo.dummy import DummyConfig
from .base import BaseAgent
import jax.numpy as jnp

class DummyAgent(BaseAgent):
    """
    Dummy agent for testing purposes.
    """
    name = "DummyAgent"
    model_names = []

    def __init__(self, obs_dim: int, act_dim: int, cfg: DummyConfig) -> None:
        super().__init__(obs_dim, act_dim, cfg)

    def train_step(self, batch, step: int):
        return {"loss": 0.0}
    
    def pretrain_step(self, batch, step: int):
        return {"loss": 0.0}
    
    def compute_statistics(self, batch):
        return {"discount": self.cfg.discount}
    
    def sample_actions(self, obs, use_behavior = False, temperature = 0, num_samples = 1, return_history = False):
        batch_size = obs.shape[0]
        action_shape = (batch_size, self.act_dim)
        actions = jnp.zeros(action_shape)
        info = {"log_prob": jnp.zeros(batch_size)}
        return actions, info
    
