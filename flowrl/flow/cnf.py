from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass, field
from flax.training.train_state import TrainState

from flowrl.functional.activation import mish
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import *

# ======= Flow Network ========

class FlowBackbone(nn.Module):
    vel_predictor: nn.Module
    time_embedding: nn.Module = None
    cond_embedding: nn.Module = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False
    ):
        if self.time_embedding is not None:
            time = self.time_embedding()(time)
            time = MLP(
                hidden_dims=[time.shape[-1], time.shape[-1]],
                activation=mish,
            )(time)
        if self.cond_embedding is not None:
            condition = self.cond_embedding()(condition, training=training)
        if condition is not None:
            inputs = jnp.concatenate([x, time, condition], axis=-1)
        else:
            inputs = jnp.concatenate([x, time], axis=-1)
        return self.vel_predictor()(inputs, training=training)

# ======= CNF ========

@dataclass
class ContinuousNormalizingFlow(Model):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
    steps: int = field(pytree_node=False, default=None)
    clip_sampler: bool = field(pytree_node=False, default=None)
    x_min: float = field(pytree_node=False, default=None)
    x_max: float = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        network: FlowBackbone,
        rng: PRNGKey,
        inputs: Sequence[jnp.ndarray],
        x_dim: int,
        steps: int,
        clip_sampler: bool=False,
        x_min: Optional[float]=None,
        x_max: Optional[float]=None,
        optimizer: Optional[optax.GradientTransformation]=None,
        clip_grad_norm: float=None
    ) -> 'ContinuousNormalizingFlow':
        ret = super().create(network, rng, inputs, optimizer, clip_grad_norm)

        return ret.replace(
            x_dim=x_dim,
            steps=steps,
            clip_sampler=clip_sampler,
            x_min=x_min,
            x_max=x_max,
        )

    def step2t(self, step: jnp.ndarray) -> jnp.ndarray:
        """Convert discrete step in [0, self.steps] to continuous time in [0, 1]."""
        t = step / self.steps
        return t

    def linear_interpolation(self, rng, x0, x1):
        rng, t_rng = jax.random.split(rng, 2)
        t = jax.random.uniform(t_rng, (*x1.shape[:-1], 1))
        xt = (1 - t) * x0 + t * x1
        vel = x1 - x0
        return rng, xt, t, vel

    def _ode_step(
        self,
        dropout_rng: PRNGKey,
        t: jnp.ndarray,
        condition: jnp.ndarray,
        xt: jnp.ndarray,
        training: bool,
        params: Optional[Param]=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Move flow forward one step with Euler method"""
        if training:
            vel = self.apply(
                {"params": params}, xt, t, condition=condition, training=training, rngs={"dropout": dropout_rng}
            )
        else:
            vel = self(xt, t, condition=condition, training=training)
        x_next = xt + vel / self.steps
        if self.clip_sampler:
            x_next = jnp.clip(x_next, self.x_min, self.x_max)
        return x_next, vel

    @partial(jax.jit, static_argnames=("training"))
    def sample(
        self,
        rng: PRNGKey,
        x0: jnp.ndarray,
        condition: Optional[jnp.ndarray]=None,
        training: bool=False,
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        t_proto = jnp.ones((*x0.shape[:-1], 1), dtype=jnp.int32)

        def fn(input, t):
            rng_, xt = input
            rng_, dropout_rng_ = jax.random.split(rng_, 2)

            x_next, vel = self._ode_step(
                dropout_rng_,
                t*t_proto,
                condition,
                xt,
                training=training,
                params=params,
            )

            return (rng_, x_next), (xt, vel)

        output, history = jax.lax.scan(fn, (rng, x0), self.step2t(jnp.arange(self.steps)), unroll=True)
        rng, x1 = output
        return rng, x1, history

# ======= Update Function ========

@jax.jit
def jit_update_flow_matching(
    rng: PRNGKey,
    model: ContinuousNormalizingFlow,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    condition: Optional[jnp.ndarray]=None,
) -> Tuple[PRNGKey, ContinuousNormalizingFlow, Metric]:
    rng, xt, t, vel = model.linear_interpolation(rng, x0, x1)

    def loss_fn(params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        vel_pred = model.apply(
            {"params": params},
            xt,
            t,
            condition=condition,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((vel_pred - vel) ** 2).mean()
        return loss, {
            "fm_loss": loss
        }

    new_model, metrics = model.apply_gradient(loss_fn)
    return rng, new_model, metrics
