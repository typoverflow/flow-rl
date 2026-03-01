from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.struct import dataclass, field
from flax.training.train_state import TrainState

from flowrl.module.model import Model
from flowrl.types import *

# ======= GenPO Flow ========

def _heun_forward_single(apply_fn, params, obs_s, x0_s, a_dim, steps, mix_para):
    """Forward Heun solver for a single (unbatched) sample.

    Internal helper for jacrev + vmap.
    """
    z = x0_s[:a_dim]
    y = x0_s[a_dim:]
    dt = 1.0 / steps

    def step_fn(carry, t_val):
        z, y = carry
        t_arr = jnp.expand_dims(t_val, axis=-1)
        v_y = apply_fn({"params": params}, y, t_arr, condition=obs_s)
        z_in = z + v_y * dt
        v_z = apply_fn({"params": params}, z_in, t_arr, condition=obs_s)
        y_in = y + v_z * dt
        z = mix_para * z_in + (1 - mix_para) * y_in
        y = mix_para * y_in + (1 - mix_para) * z
        return (z, y), None

    ts = jnp.arange(steps, dtype=jnp.float32) / steps
    (z_f, y_f), _ = jax.lax.scan(step_fn, (z, y), ts)
    return jnp.concatenate([z_f, y_f], axis=-1)


def _heun_inverse_single(apply_fn, params, obs_s, x1_s, a_dim, steps, mix_para):
    """Inverse Heun solver for a single (unbatched) sample.

    Internal helper for jacrev + vmap.
    """
    z = x1_s[:a_dim]
    y = x1_s[a_dim:]
    dt = 1.0 / steps

    def step_fn(carry, t_val):
        z, y = carry
        t_arr = jnp.expand_dims(t_val, axis=-1)
        # Unmix (reverse of: z_new = p*z_in + (1-p)*y_in; y_new = p*y_in + (1-p)*z_new)
        y_in = (y - (1 - mix_para) * z) / mix_para
        z_in = (z - (1 - mix_para) * y_in) / mix_para
        # Undo y step
        v_z = apply_fn({"params": params}, z_in, t_arr, condition=obs_s)
        y = y_in - v_z * dt
        # Undo z step
        v_y = apply_fn({"params": params}, y, t_arr, condition=obs_s)
        z = z_in - v_y * dt
        return (z, y), None

    ts = jnp.arange(steps - 1, -1, -1, dtype=jnp.float32) / steps
    (z0, y0), _ = jax.lax.scan(step_fn, (z, y), ts)
    return jnp.concatenate([z0, y0], axis=-1)


@dataclass
class GenPOFlow(Model):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    a_dim: int = field(pytree_node=False, default=None)
    steps: int = field(pytree_node=False, default=None)
    mix_para: float = field(pytree_node=False, default=None)

    @classmethod
    def create(cls, network, rng, inputs, a_dim, steps, mix_para,
               optimizer=None, clip_grad_norm=None):
        ret = super().create(network, rng, inputs, optimizer, clip_grad_norm)
        return ret.replace(a_dim=a_dim, steps=steps, mix_para=mix_para)

    @partial(jax.jit, static_argnames=("training",))
    def forward(self, obs, x0, training=False, params=None):
        """Batched forward Heun solver: noise -> augmented action."""
        if params is None:
            params = self.params
        z = x0[:, :self.a_dim]
        y = x0[:, self.a_dim:]
        dt = 1.0 / self.steps
        t_proto = jnp.ones((x0.shape[0], 1))

        def step_fn(carry, t_val):
            z, y = carry
            t_arr = t_val * t_proto
            v_y = self.apply({"params": params}, y, t_arr, condition=obs)
            z_in = z + v_y * dt
            v_z = self.apply({"params": params}, z_in, t_arr, condition=obs)
            y_in = y + v_z * dt
            z = self.mix_para * z_in + (1 - self.mix_para) * y_in
            y = self.mix_para * y_in + (1 - self.mix_para) * z
            return (z, y), None

        ts = jnp.arange(self.steps, dtype=jnp.float32) / self.steps
        (z_f, y_f), _ = jax.lax.scan(step_fn, (z, y), ts)
        return jnp.concatenate([z_f, y_f], axis=-1)

    def inverse(self, obs, x1, params=None):
        """Batched inverse Heun solver: augmented action -> noise."""
        if params is None:
            params = self.params
        fn = partial(_heun_inverse_single, self.state.apply_fn, params,
                     a_dim=self.a_dim, steps=self.steps, mix_para=self.mix_para)
        return jax.vmap(fn)(obs, x1)

    def log_prob(self, obs, x0, params=None):
        """Compute log probability via forward Jacobian (used during sampling).

        log π(x1) = log N(x0; 0, I) - log|det J_forward(x0)|
        """
        if params is None:
            params = self.params
        apply_fn = self.state.apply_fn
        aug_dim = 2 * self.a_dim

        def forward_single(x0_s, obs_s):
            return _heun_forward_single(apply_fn, params, obs_s, x0_s,
                                        self.a_dim, self.steps, self.mix_para)

        J = jax.vmap(jax.jacrev(forward_single, argnums=0))(x0, obs)
        _, logabsdet = jnp.linalg.slogdet(J)
        log_p_x0 = -0.5 * jnp.sum(x0 ** 2, axis=-1) - 0.5 * aug_dim * jnp.log(2 * jnp.pi)
        return log_p_x0 - logabsdet

    def log_prob_via_inverse(self, obs, x1, params=None):
        """Compute log probability via inverse Jacobian (used during training).

        log π(x1) = log N(f^{-1}(x1); 0, I) + log|det J_inverse(x1)|
        """
        if params is None:
            params = self.params
        apply_fn = self.state.apply_fn
        aug_dim = 2 * self.a_dim

        def inverse_single(x1_s, obs_s):
            return _heun_inverse_single(apply_fn, params, obs_s, x1_s,
                                        self.a_dim, self.steps, self.mix_para)

        x0 = jax.vmap(inverse_single)(x1, obs)
        J = jax.vmap(jax.jacrev(inverse_single, argnums=0))(x1, obs)
        _, logabsdet = jnp.linalg.slogdet(J)
        log_p_x0 = -0.5 * jnp.sum(x0 ** 2, axis=-1) - 0.5 * aug_dim * jnp.log(2 * jnp.pi)
        return log_p_x0 + logabsdet
