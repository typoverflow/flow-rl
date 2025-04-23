from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass, field
from flax.training.train_state import TrainState

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
        s: jnp.ndarray,  # s = [obs, cond] if it's conditional
        a: jnp.ndarray,
        time: jnp.ndarray,
        training: bool = False
    ):
        if self.cond_embedding is not None:
            # last dim gives the class token
            embed_feature = self.cond_embedding()(s[:, -1], training=training)  # gives the shape of (B,) array
            s = jnp.concatenate([s[:, :-1], embed_feature], axis=-1)
        if self.time_embedding is not None:
            time = self.time_embedding()(time)
        inputs = jnp.concatenate([s, a, time], axis=-1)
        x = self.vel_predictor()(inputs, training=training)
        return x

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
        obs: jnp.ndarray,
        action: jnp.ndarray,
        training: bool,
        params: Optional[Param]=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Move flow forward one step with Euler method"""
        if training:
            vel = self.apply(
                {"params": params}, obs, action, t, training=training, rngs={"dropout": dropout_rng}
            )
        else:
            vel = self(obs, action, t, training=training)
        action = action + vel / self.steps
        if self.clip_sampler:
            action = jnp.clip(action, self.x_min, self.x_max)
        return action, vel

    @partial(jax.jit, static_argnames=("training", "sample_xt"))
    def onestep_sample(
        self,
        rng: PRNGKey,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        training: bool,
        sample_xt: bool,
        t: Optional[jnp.ndarray]=None,
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        raise NotImplementedError()

    @partial(jax.jit, static_argnames=("training", "num_samples"))
    def sample(
        self,
        rng: PRNGKey,
        x0: jnp.ndarray,
        obs: jnp.ndarray,
        training: bool,
        num_samples: Optional[int]=None,
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        if num_samples is not None:
            obs_use = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
        else:
            obs_use = obs
        t_proto = jnp.ones((*obs.shape[:-1], 1), dtype=jnp.int32)

        def fn(input, t):
            rng_, xt = input
            rng_, dropout_rng_ = jax.random.split(rng_, 2)

            x_next, vel = self._ode_step(
                dropout_rng_,
                t*t_proto,
                obs_use,
                xt,
                training=training,
                params=params,
            )

            return (rng_, x_next), (xt, vel)

        output, history = jax.lax.scan(fn, (rng, x0), self.step2t(jnp.arange(self.steps)), unroll=True)
        rng, action = output
        return rng, action, history

# ======= Update Function ========

@jax.jit
def jit_update_flow_matching(
    rng: PRNGKey,
    model: ContinuousNormalizingFlow,
    x0: jnp.ndarray,
    batch: Batch,
) -> Tuple[PRNGKey, ContinuousNormalizingFlow, Metric]:
    x1 = batch.action
    rng, xt, t, vel = model.linear_interpolation(rng, x0, x1)

    def loss_fn(params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        vel_pred = model.apply(
            {"params": params},
            batch.obs,
            xt,
            t,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((vel_pred - vel) ** 2).mean()
        return loss, {
            "fm_loss": loss
        }

    new_model, metrics = model.apply_gradient(loss_fn)
    return rng, new_model, metrics
