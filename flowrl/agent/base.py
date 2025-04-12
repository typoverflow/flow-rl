from typing import Dict, Tuple
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as orbax

from flowrl.config.d4rl.algo.base import BaseAlgoConfig
from flowrl.types import Batch


class BaseAgent():
    """
    Base class for all agents.
    """
    name = "BaseAgent"
    model_names = []

    def __init__(self, obs_dim: int, act_dim: int, cfg: BaseAlgoConfig) -> None:
        """
        Initialize the agent.

        Args:
            obs_dim (int): The dimension of the observation space.
            act_dim (int): The dimension of the action space.
            cfg (BaseAlgoConfig): The configuration object for the algorithm.
        """
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def prepare_training(self) -> None:
        """
        Prepare the agent for training.
        This function is called after pretrain (or load pretrained checkpoint) and before training.
        It is used to set up the agent's state, such as copying pretrained parameters to the training model.
        """
        pass

    def compute_statistics(self, batch: Batch) -> Dict:
        """
        Evaluate the agent on a batch of data.
        Args:
            batch (Batch): The batch of data to compute statistics on.
        Returns:
            Dict: The information dictionary containing statistics.
        """
        return None

    def train_step(self, batch: Batch, step: int) -> Dict:
        """
        Perform a training step on the agent.

        Args:
            batch (Batch): The batch of data to train on.
            step (int): The current training step.

        Returns:
            Dict: The information dictionary containing training statistics.
        """
        raise NotImplementedError("train_step not implemented for this agent")
    
    def pretrain_step(self, batch: Batch, step: int) -> Dict:
        """
        Perform a pretraining step on the agent.
        Args:
            batch (Batch): The batch of data to pretrain on.
            step (int): The current pretraining step.

        Returns:
            Dict: The information dictionary containing pretraining statistics.
        """
        raise NotImplementedError("pretrain_step not implemented for this agent")
    
    def sample_actions(
            self,
            obs: jnp.ndarray,
            use_behavior: bool = False,
            temperature: float = 0.0,
            num_samples: int = 1,
            return_history: bool = False,
        ) -> Tuple[jnp.ndarray, Dict[str, any]]:
        """
        Sample actions from the agent's policy.

        Args:
            observations (jnp.ndarray): The observations to sample actions from. Shape should be
                                        (batch_size, obs_dim).
            use_behavior (bool): Whether to use the behavior policy for sampling.
            temperature (float): The temperature for sampling.
            num_samples (int): The number of actions to sample.
            return_history (bool): Whether to return the history of sampling.

        Returns:
            actions (jnp.ndarray): The sampled actions. Shape should be (batch_size, num_samples, action_dim).
            info (Dict[str, any]): Additional information about the sampling process.
                                    
        """
        raise NotImplementedError("sample_actions not implemented for this agent")
    
    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.
        Args:
            path (str): The path to save the agent's state.
        """
        ckpt = {name: nnx.state(getattr(self, name)) for name in self.model_names}
        checkpointer = orbax.PyTreeCheckpointer()
        checkpointer.save(path, ckpt)
        
    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.
        Args:
            path (str): The path to load the agent's state from.
        """
        checkpointer = orbax.PyTreeCheckpointer()
        ckpt = {name: nnx.state(getattr(self, name)) for name in self.model_names}
        ckpt = checkpointer.restore(path, item=ckpt)
        for name in self.model_names:
            nnx.update(getattr(self, name), ckpt[name])
