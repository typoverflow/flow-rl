from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode, dataclass, field
from flax.training.train_state import TrainState

from flowrl.module.model import Model
from flowrl.types import *


class DummyModel(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray = None):
        return

def exponential_schedule(init, decay, steps):
    return init * (decay ** jnp.arange(steps))

def polynomial_schedule(init, final, power, steps):
    return (init - final) * (1 - jnp.arange(steps) / (steps - 1)) ** power + final


@dataclass
class IBCLangevinDynamics(Model):
    """
    Langevin Dynamics Sampler from Implicit Behavior Cloning (https://arxiv.org/abs/2109.00137)
    """
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
    steps: int = field(pytree_node=False, default=None)
    schedule: str = field(pytree_node=False, default=None)
    stepsize_init: float = field(pytree_node=False, default=None)
    stepsize_final: float = field(pytree_node=False, default=None)
    stepsize_decay: float = field(pytree_node=False, default=None)
    stepsize_power: float = field(pytree_node=False, default=None)
    noise_scale: float = field(pytree_node=False, default=None)
    grad_clip: float | None = field(pytree_node=False, default=None)
    drift_clip: float | None = field(pytree_node=False, default=None)
    margin_clip: float | None = field(pytree_node=False, default=None)
    x_min: float = field(pytree_node=False, default=None)
    x_max: float = field(pytree_node=False, default=None)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        x_dim: int,
        steps: int = 100,
        schedule: str = "polynomial",
        stepsize_init: float = 1e-1,
        stepsize_final: float = 1e-5,
        stepsize_decay: float = 0.8,
        stepsize_power: float = 2.0,
        noise_scale: float = 1.0,
        grad_clip: float | None = None,
        drift_clip: float | None = None,
        margin_clip: float | None = None,
    ) -> 'IBCLangevinDynamics':
        network = DummyModel()
        inputs = ()
        ret = super().create(network, rng, inputs)

        return ret.replace(
            x_dim=x_dim,
            steps=steps,
            schedule=schedule,
            stepsize_init=stepsize_init,
            stepsize_final=stepsize_final,
            stepsize_decay=stepsize_decay,
            stepsize_power=stepsize_power,
            noise_scale=noise_scale,
            grad_clip=grad_clip,
            drift_clip=drift_clip,
            margin_clip=margin_clip,
        )

    @partial(jax.jit, static_argnames=("model_fn"))
    def sample(
        self,
        rng: PRNGKey,
        model_fn: Callable,
        xT: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        if self.schedule == "polynomial":
            stepsizes = polynomial_schedule(
                self.stepsize_init,
                self.stepsize_final,
                self.stepsize_power,
                self.steps,
            )
        else:
            stepsizes = exponential_schedule(
                self.stepsize_init,
                self.stepsize_decay,
                self.steps,
            )
        t_proto = jnp.ones((*xT.shape[:-1], 1), dtype=jnp.int32)

        def fn(input_tuple, i):
            rng_, xt = input_tuple
            rng_, dropout_rng_, key_ = jax.random.split(rng_, 3)

            energy, q_grad = model_fn(xt, t_proto * i, condition=condition)
            if self.grad_clip is not None:
                q_grad = jnp.clip(q_grad, -self.grad_clip, self.grad_clip)

            drift = stepsizes[i] * (
                0.5 * q_grad +\
                jax.random.normal(key_, xt.shape) * self.noise_scale
            )
            if self.drift_clip is not None:
                drift = jnp.clip(drift, -self.drift_clip, self.drift_clip)
            xt_1 = xt + drift
            if self.margin_clip is not None:
                xt_1 = jnp.clip(xt_1, -self.margin_clip, self.margin_clip)
            return (rng_, xt_1), (xt, q_grad, energy)

        output, history = jax.lax.scan(fn, (rng, xT), jnp.arange(self.steps), unroll=True)
        rng, action = output
        return rng, action, history
