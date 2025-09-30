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
    rff_dim = None
    linear: bool = False

    def setup(self):
        # TODO: PositionalFeature?
        # jax nn mish vs flax nn
        # no need to do any device work right
        self.mlp_t = nn.Sequential(
            PositionalEmbedding(128), nn.Dense(256), mish, nn.Dense(128)
        )
        self.mlp_phi = ResidualMLP(
            self.phi_hidden_dims, self.feature_dim, activation=mish, dropout=None
        )
        self.mlp_mu = ResidualMLP(
            self.mu_hidden_dims, self.feature_dim, activation=mish, dropout=None
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
