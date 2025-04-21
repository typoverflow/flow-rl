from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass, field
from flax.training.train_state import TrainState

from flowrl.functional.ema import ema_update
from flowrl.module.model import Model
from flowrl.types import *

# ======= Velocity Field Network ========

class VelocityField(nn.Module):
    backbone: nn.Module

    @nn.compact
    def __call__(
        self,
        s: jnp.ndarray,  # s = [obs, cond] if it's conditional
        a: jnp.ndarray,
        time: jnp.ndarray,
        training: bool = False
    ):
        inputs = jnp.concatenate([s, a, time], axis=-1)
        x = self.backbone(inputs, training=training)
        return x

class OneStepTransform(nn.Module):
    backbone: nn.Module

    @nn.compact
    def __call__(
        self,
        s: jnp.ndarray,
        a: jnp.ndarray,
        training: bool = False
    ):
        inputs = jnp.concatenate([s, a], axis=-1)
        x = self.backbone(inputs, training=training)
        return x

# ======= Flow Matching ========

@dataclass
class FlowMatching(Model):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    steps: int = field(pytree_node=False, default=None)
    x_dim: int = field(pytree_node=False, default=None)
    clip_sampler: bool = field(pytree_node=False, default=None)
    x_min: float = field(pytree_node=False, default=None)
    x_max: float = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        network: VelocityField,
        rng: PRNGKey,
        inputs: Sequence[jnp.ndarray],
        steps: int,
        x_dim: int,
        clip_sampler: bool=False,
        x_min: Optional[float]=None,
        x_max: Optional[float]=None,
        optimizer: Optional[optax.GradientTransformation]=None,
        clip_grad_norm: float=None
    ) -> 'FlowMatching':
        ret = super().create(network, rng, inputs, optimizer, clip_grad_norm)

        return ret.replace(
            steps=steps,
            x_dim=x_dim,
            clip_sampler=clip_sampler,
            x_min=x_min,
            x_max=x_max,
        )
    
    def step2t(self, step: jnp.ndarray) -> jnp.ndarray:
        """Convert discrete step in [0, self.steps] to continuous time in [0, 1]."""
        t = step / self.steps
        return t


    def add_noise(self, key, x1):
        B, _ = x1.shape
        t_key, noise_key = jax.random.split(key)
        t = jax.random.uniform(t_key, (B, 1))
        x0 = jax.random.normal(noise_key, x1.shape)
        xt = (1 - t) * x0 + t * x1
        vel = x1 - x0
        return xt, t, vel
    
    def _onestep_sample(
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
        rng, sample_rng, dropout_rng = jax.random.split(rng, 3)
        if sample_xt:
            x0 = action
            xt, t, _ = self.add_noise(sample_rng, x0)
        else:
            t = t
            xt = action
        x_next, vel = self._onestep_sample(
            dropout_rng,
            t,
            obs,
            xt,
            training=training,
            params=params,
        )
        return rng, x_next, xt, step+1, step, vel
        

    @partial(jax.jit, static_argnames=("training", "sample_noise"))
    def sample(
        self,
        key: PRNGKey,
        obs: jnp.ndarray,
        training: bool,
        sample_noise: bool = True,
        noise: Optional[jnp.ndarray] = None,
        params: Optional[Param]=None,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        B, _ = obs.shape
        t_proto = jnp.ones((B, 1), dtype=jnp.int32)
        x0_rng, dropout_rng = jax.random.split(key)
        if sample_noise:
            x0 = jax.random.normal(x0_rng, (B, self.x_dim))
        else:
            x0 = noise

        def fn(input, t):
            rng_, xt = input
            rng_, dropout_rng_ = jax.random.split(rng_, 2)
            
            x_next, vel = self._onestep_sample(
                dropout_rng_,
                t*t_proto,
                obs,
                xt,
                training=training,
                params=params,
            )

            return (rng_, x_next), (xt, vel)

        output, history = jax.lax.scan(fn, (dropout_rng, x0), self.step2t(jnp.arange(self.steps)), unroll=True)
        _, action = output
        return action, history

# ======= Update Function ========

@jax.jit
def jit_update_flow_matching(
    key: PRNGKey,
    model: FlowMatching,
    batch: Batch,
) -> Tuple[PRNGKey, FlowMatching, Metric]:
    x0 = batch.action
    xt, t, vel = model.add_noise(key, x0)

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
    return new_model, metrics
