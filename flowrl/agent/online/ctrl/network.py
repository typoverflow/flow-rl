import flax.linen as nn
import jax.numpy as jnp

from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.module.mlp import ResidualMLP
from flowrl.module.critic import RffReward
from flowrl.functional.activation import mish
from flowrl.types import Sequence


class FactorizedNCE(nn.Module):
    obs_dim: int
    action_dim: int
    feature_dim: int
    phi_hidden_dims: Sequence[int]
    mu_hidden_dims: Sequence[int]
    reward_hidden_dims: Sequence[int]
    rff_dim: int = 0
    num_noises: int = 0
    linear: bool = False
    ranking: bool = False

    def setup(self):
        # TODO: PositionalFeature?
        # jax nn mish vs flax nn
        # no need to do any device work right
        self.mlp_t = nn.Sequential(
            [PositionalEmbedding(128), nn.Dense(256), mish, nn.Dense(128)]
        )
        self.mlp_phi = ResidualMLP(
            self.phi_hidden_dims, self.feature_dim, multiplier=1,activation=mish, dropout=None
        )
        self.mlp_mu = ResidualMLP(
            self.mu_hidden_dims, self.feature_dim, multiplier=1, activation=mish, dropout=None
        )

        self.reward = RffReward(
            self.feature_dim,
            self.reward_hidden_dims,
            linear=self.linear,
            rff_dim=self.rff_dim,
        )

    def forward_phi(self, s, a):
        x = jnp.concat([s, a], axis=-1)
        x = self.mlp_phi(x)
        return x

    def forward_mu(self, sp, t=None):
        if t is not None:
            t_ff = self.mlp_t(t)
            sp = jnp.concat([sp, t_ff], axis=-1)
        sp = self.mlp_mu(sp)
        return jnp.tanh(sp)

    def forward_reward(self, x: jnp.ndarray):  # for z_phi
        return self.reward(x)

    def get_normalizer(self, num_noises: int):
        N = max(num_noises, 1)
        if self.ranking:
            return jnp.zeros((N,), dtype=jnp.float32)
        return self.param("normalizer", lambda key: jnp.zeros((N,), jnp.float32))

    def __call__(
        self,
        s,
        a,
        sp,
    ):
        z_phi = self.forward_phi(s, a)
        _ = self.forward_reward(z_phi)

        B = s.shape[0]
        if self.num_noises > 0:
            dummy_t = jnp.zeros((self.num_noises, B, 1), dtype=s.dtype)
            dummy_xt = jnp.zeros((self.num_noises, B, sp.shape[-1]), dtype=sp.dtype)
            _ = self.forward_mu(dummy_xt, dummy_t)
        else:
            dummy_xt = jnp.expand_dims(sp, 0)
            _ = self.forward_mu(dummy_xt, None)

        _ = self.get_normalizer(self.num_noises)

        return jnp.array(0.0, dtype=s.dtype)
