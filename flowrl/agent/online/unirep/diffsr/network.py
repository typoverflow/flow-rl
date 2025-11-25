from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp

from flowrl.flow.ddpm import get_noise_schedule
from flowrl.functional.activation import l2_normalize, mish
from flowrl.module.mlp import MLP, ResidualMLP
from flowrl.module.rff import RffReward
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import *
from flowrl.types import Sequence


class ResidualCritic(nn.Module):
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, ft: jnp.ndarray, t: jnp.ndarray, training: bool=False):
        t_ff = self.time_embedding()(t)
        t_ff = MLP(
            hidden_dims=[t_ff.shape[-1], t_ff.shape[-1]],
            activation=mish,
        )(t_ff)
        x = jnp.concatenate([item for item in [ft, ] if item is not None], axis=-1)
        x = ResidualMLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            multiplier=1,
            activation=self.activation,
            layer_norm=True,
        )(x, training)
        return x

class FactorizedDDPM(nn.Module):
    obs_dim: int
    action_dim: int
    feature_dim: int
    embed_dim: int
    phi_hidden_dims: Sequence[int]
    mu_hidden_dims: Sequence[int]
    reward_hidden_dims: Sequence[int]
    rff_dim: int
    num_noises: int

    def setup(self):
        self.mlp_t1 = nn.Sequential([
            PositionalEmbedding(self.embed_dim),
            nn.Dense(2*self.embed_dim),
            mish,
            nn.Dense(self.embed_dim)
        ])
        self.mlp_t2 = nn.Sequential([
            PositionalEmbedding(self.embed_dim),
            nn.Dense(2*self.embed_dim),
            mish,
            nn.Dense(self.embed_dim)
        ])
        self.mlp_s = nn.Sequential([
            nn.Dense(self.embed_dim*2),
            mish,
            nn.Dense(self.embed_dim)
        ])
        self.mlp_a = nn.Sequential([
            nn.Dense(self.embed_dim*2),
            mish,
            nn.Dense(self.embed_dim)
        ])
        self.mlp_phi = ResidualMLP(
            self.phi_hidden_dims,
            self.feature_dim,
            multiplier=1,
            activation=mish,
            layer_norm=True,
            dropout=None,
        )
        self.mlp_mu = ResidualMLP(
            self.mu_hidden_dims,
            self.feature_dim*self.obs_dim,
            multiplier=1,
            activation=mish,
            layer_norm=True,
            dropout=None,
        )
        self.reward = RffReward(
            self.feature_dim,
            self.reward_hidden_dims,
            rff_dim=self.rff_dim,
        )
        betas, alphas, alphabars = get_noise_schedule("vp", self.num_noises)
        alphabars_prev = jnp.pad(alphabars[:-1], (1, 0), constant_values=1.0)
        self.alphabars = alphabars

    def forward_phi(self, s, a, t):
        s = self.mlp_s(s)
        a = self.mlp_a(a)
        t_ff = self.mlp_t1(t)
        x = jnp.concat([s, a, t_ff], axis=-1)
        x = self.mlp_phi(x)
        return x

    def forward_mu(self, sp, t):
        t = self.mlp_t2(t)
        x = jnp.concat([sp, t], axis=-1)
        x = self.mlp_mu(x)
        return x.reshape(-1, self.feature_dim, self.obs_dim)

    def forward_reward(self, x: jnp.ndarray):
        return self.reward(x)

    def __call__(self, rng, s, a, sp, training: bool=False):
        rng, t1_rng, eps1_rng, t2_rng, eps2_rng = jax.random.split(rng, 5)
        t1 = jax.random.randint(t1_rng, (s.shape[0], 1), 0, self.num_noises+1)
        t2 = jax.random.randint(t2_rng, (s.shape[0], 1), 0, self.num_noises+1)
        eps1 = jax.random.normal(eps1_rng, a.shape)
        eps2 = jax.random.normal(eps2_rng, sp.shape)
        at = jnp.sqrt(self.alphabars[t1]) * a + jnp.sqrt(1-self.alphabars[t1]) * eps1
        spt = jnp.sqrt(self.alphabars[t2]) * sp + jnp.sqrt(1-self.alphabars[t2]) * eps2
        z_phi = self.forward_phi(s, at, t1)
        z_mu = self.forward_mu(spt, t2)
        eps_pred = jax.lax.batch_matmul(z_phi[..., jnp.newaxis, :], z_mu)[..., 0, :]
        r_pred = self.forward_reward(z_phi)
        return eps2, eps_pred, r_pred


@partial(jax.jit, static_argnames=("reward_coef"))
def update_factorized_ddpm(
    rng: PRNGKey,
    ddpm: FactorizedDDPM,
    batch: Batch,
    reward_coef: float,
) -> Tuple[PRNGKey, FactorizedDDPM, Metric]:
    B = batch.obs.shape[0]
    rng, update_rng = jax.random.split(rng)
    def loss_fn(ddpm_params: Param, dropout_rng: PRNGKey):
        eps, eps_pred, r_pred = ddpm.apply(
            {"params": ddpm_params},
            update_rng,
            batch.obs,
            batch.action,
            batch.next_obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        ddpm_loss = ((eps_pred - eps) ** 2).mean()
        reward_loss = ((r_pred - batch.reward) ** 2).mean()
        loss = ddpm_loss + reward_coef * reward_loss
        return loss, {
            "loss/ddpm_loss": ddpm_loss,
            "loss/reward_loss": reward_loss,
            "misc/sp0_mean": batch.next_obs.mean(),
            "misc/sp0_std": batch.next_obs.std(axis=0).mean(),
            "misc/eps_mean": eps_pred.mean(),
            "misc/reward_mean": r_pred.mean(),
        }

    new_ddpm, metrics = ddpm.apply_gradient(loss_fn)
    return rng, new_ddpm, metrics
