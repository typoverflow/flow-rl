from abc import ABC
from typing import Optional, Sequence, Tuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass, PyTreeNode
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
    dropout_rng: PRNGKey
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
        init_rng, dropout_rng = jax.random.split(rng)
        params = network.init(init_rng, *inputs)  # params = {"params": ...}

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
            params=params["params"],
            tx=optimizer
        )
        return cls(
            state=state,
            dropout_rng=dropout_rng
        )

    def __call__(self, *args, **kwargs):
        # the network defined by the jax nn.Model should be used by apply function with {'params': P} and other ..
        dropout_rng, next_dropout_rng = jax.random.split(self.dropout_rng)
        out = self.state.apply_fn(
            {'params': self.state.params},
            *args,
            **kwargs,
            training=False, # training is set to False, all training calls should be done with `apply`
            rngs={'dropout': dropout_rng}
        )
        return self.replace(
            dropout_rng=next_dropout_rng,
        ), out
    
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
