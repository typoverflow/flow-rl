from abc import ABC
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import PyTreeNode, dataclass, field
from flax.training.train_state import TrainState

from flowrl.types import Metric, PRNGKey


def empty_optimizer() -> optax.GradientTransformation:
    """Returns an empty optimizer, which does not update any parameters."""
    def init_fn(params):
        return True # bool is a valid opt_state
    def update_fn(grads, state, params=None):
        raise ValueError("Empty optimizer does not update any parameters.")
    return optax.GradientTransformation(init_fn, update_fn)

@dataclass
class Model(PyTreeNode, ABC):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    """

    Model is initialized with Model.create() method
    """

    @classmethod
    def create(
        cls,
        network: nn.Module,
        rng: PRNGKey,  # PRNGKey
        inputs: Sequence[jnp.ndarray],  # sample of inputs
        optimizer: Optional[optax.GradientTransformation] = None,
        clip_grad_norm: float = None
    ) -> 'Model':
        param_rng, dropout_rng = jax.random.split(rng)
        params = network.init(param_rng, *inputs)  # params = {"params": ...}

        if optimizer is not None:
            if clip_grad_norm:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(max_norm=clip_grad_norm),
                    optimizer
                )
        else:
            optimizer = empty_optimizer()

        state = TrainState.create(
            apply_fn=network.apply,
            params=params.get("params", {}),
            tx=optimizer
        )
        return cls(
            state=state,
            dropout_rng=dropout_rng,
        )

    @property
    def params(self):
        return self.state.params

    def __call__(self, *args, **kwargs):
        # the network defined by the jax nn.Model should be used by apply function with {'params': P} and other ..
        out = self.state.apply_fn(
            {'params': self.state.params},
            *args,
            **kwargs,
        )
        return out

    def apply(self, *args, **kwargs):
        return self.state.apply_fn(*args,**kwargs)

    def apply_gradient(self, loss_fn) -> Tuple['Model', Metric]:
        grad_fn = jax.grad(loss_fn, has_aux=True)  # here auxiliary data is just the info dict
        dropout_rng, next_dropout_rng = jax.random.split(self.dropout_rng)
        grads, info = grad_fn(self.state.params, dropout_rng)
        state = self.state.apply_gradients(grads=grads)
        return self.replace(
            state=state,
            dropout_rng=next_dropout_rng
        ), info

    def reset_optim(self, optimizer: optax.GradientTransformation) -> 'Model':
        """Reset the optimizer to a new one."""
        new_state = TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,
            tx=optimizer
        )
        return self.replace(state=new_state)
