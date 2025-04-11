from typing import Dict
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as orbax

from flowrl.dataset import Batch


class BaseAgent():
    """
    Base class for all agents.
    """
    name = "BaseAgent"
    model_names = []

    def train_step(self, batch: Batch) -> Dict:
        """
        Perform a training step on the agent.

        Args:
            batch (Batch): The batch of data to train on.

        Returns:
            Dict: The information dictionary containing training statistics.
        """
        raise NotImplementedError("train_step not implemented for this agent")
    
    def pretrain_step(self, batch: Batch) -> Dict:
        """
        Perform a pretraining step on the agent.
        Args:
            batch (Batch): The batch of data to pretrain on.
        Returns:
            Dict: The information dictionary containing pretraining statistics.
        """
        raise NotImplementedError("pretrain_step not implemented for this agent")
    
    def sample_actions(
            self,
            obs: jnp.ndarray,
            temperature: float = 0.0,
            num_actions: int = 1,
            return_history: bool = False,
        ) -> Dict:
        """
        Sample actions from the agent's policy.

        Args:
            observations (jnp.ndarray): The observations to sample actions from.
            temperature (float): The temperature for sampling.
            num_actions (int): The number of actions to sample.
            return_history (bool): Whether to return the history of sampling.

        Returns:
            Dict[str, jnp.ndarray]: A dictionary containing the sampled actions and
                                    any additional information.
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
