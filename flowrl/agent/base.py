import os
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax
from flax.training import orbax_utils
from flax.training.train_state import TrainState

from flowrl.config.offline.d4rl.algo.base import BaseAlgoConfig
from flowrl.module.model import Model
from flowrl.types import Batch, Metric


class BaseAgent():
    """
    Base class for all agents.
    """
    name = "BaseAgent"
    model_names: List[str] = []

    def __init__(self, obs_dim: int, act_dim: int, cfg: BaseAlgoConfig, seed: int) -> None:
        """
        Initialize the agent.

        Args:
            obs_dim (int): The dimension of the observation space.
            act_dim (int): The dimension of the action space.
            cfg (BaseAlgoConfig): The configuration object for the algorithm.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.cfg = cfg
        self.rng = jax.random.PRNGKey(seed)

    def prepare_training(self) -> None:
        """
        Prepare the agent for training.
        This function is called after pretrain (or load pretrained checkpoint) and before training.
        It is used to set up the agent's state, such as copying pretrained parameters to the training model.
        """
        pass

    def compute_statistics(self, batch: Batch) -> Metric:
        """
        Evaluate the agent on a batch of data.
        Args:
            batch (Batch): The batch of data to compute statistics on.
        Returns:
            Metric: The information dictionary containing statistics.
        """
        return None

    def train_step(self, batch: Batch, step: int) -> Metric:
        """
        Perform a training step on the agent.

        Args:
            batch (Batch): The batch of data to train on.
            step (int): The current training step.

        Returns:
            Metric: The information dictionary containing training statistics.
        """
        raise NotImplementedError("train_step not implemented for this agent")

    def pretrain_step(self, batch: Batch, step: int) -> Metric:
        """
        Perform a pretraining step on the agent.
        Args:
            batch (Batch): The batch of data to pretrain on.
            step (int): The current pretraining step.

        Returns:
            Metric: The information dictionary containing pretraining statistics.
        """
        raise NotImplementedError("pretrain_step not implemented for this agent")

    def sample_actions(
            self,
            obs: jnp.ndarray,
            deterministic: bool = True,
            num_samples: int = 1,
        ) -> Tuple[jnp.ndarray, Metric]:
        """
        Sample actions from the agent's policy.

        Args:
            obs (jnp.ndarray): The observations to sample actions from. Shape should be
                                        (batch_size, obs_dim).
            deterministic (bool): Whether to sample actions deterministically or stochastically.
            num_samples (int): The number of actions to sample.

        Returns:
            actions (jnp.ndarray): The sampled actions. Shape should be (batch_size, num_samples, action_dim).
            info (Metric): Additional information about the sampling process.

        """
        raise NotImplementedError("sample_actions not implemented for this agent")

    @property
    def saved_model_names(self) -> List[str]:
        """
        Get the names of the models to be saved or loaded. Default is all models.
        Agent can override this method to specify which models to save or load,
        conditional on the agent's configuration, training stage, etc.
        Returns:
            List[str]: The names of the models to be saved.
        """
        return self.model_names

    def save(self, path: str) -> None:
        """
        Save the agent's state to a file.
        Args:
            path (str): The path to save the agent's state.
        """
        ckpt: Dict[str, TrainState] = {name: getattr(self, name).state for name in self.saved_model_names}
        checkpointer = orbax.PyTreeCheckpointer()
        # save_args = orbax_utils.save_args_from_target(ckpt)
        # checkpointer.save(os.path.join(os.getcwd(), path), ckpt, save_args=save_args)
        checkpointer.save(os.path.join(os.getcwd(), path), ckpt)

    def load(self, path: str) -> None:
        """
        Load the agent's state from a file.
        Args:
            path (str): The path to load the agent's state from.
        """
        checkpointer = orbax.PyTreeCheckpointer()
        ckpt = checkpointer.restore(os.path.join(os.getcwd(), path))
        for name in self.saved_model_names:
            model: Model = getattr(self, name)
            new_state = model.state.replace(**ckpt[name])
            setattr(self, name, model.replace(state=new_state))
