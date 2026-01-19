from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

import flowrl.module.initialization as init
from flowrl.flow.ddpm import get_noise_schedule
from flowrl.functional.activation import mish
from flowrl.module.mlp import MLP, ResidualMLP
from flowrl.module.rff import RffReward
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import *
from flowrl.types import Sequence


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
        MLP_torch_init = partial(MLP, kernel_init=init.pytorch_kernel_init, bias_init=init.pytorch_bias_init)
        self.mlp_t = nn.Sequential([
            PositionalEmbedding(self.embed_dim),
            MLP_torch_init(output_dim=2*self.embed_dim),
            mish,
            MLP_torch_init(output_dim=self.embed_dim)
        ])
        # self.mlp_s = nn.Sequential([
        #     MLP_torch_init(output_dim=self.embed_dim*2),
        #     mish,
        #     MLP_torch_init(output_dim=self.embed_dim)
        # ])
        # self.mlp_a = nn.Sequential([
        #     MLP_torch_init(output_dim=self.embed_dim*2),
        #     mish,
        #     MLP_torch_init(output_dim=self.embed_dim)
        # ])
        self.mlp_phi = ResidualMLP(
            self.phi_hidden_dims,
            self.feature_dim,
            multiplier=1,
            activation=mish,
            layer_norm=True,
            dropout=None,
            kernel_init=init.pytorch_kernel_init,
            bias_init=init.pytorch_bias_init,
        )
        self.mlp_mu = ResidualMLP(
            self.mu_hidden_dims,
            self.feature_dim*self.obs_dim,
            multiplier=1,
            activation=mish,
            layer_norm=True,
            dropout=None,
            kernel_init=init.pytorch_kernel_init,
            bias_init=init.pytorch_bias_init,
        )
        self.reward = RffReward(
            self.feature_dim,
            self.reward_hidden_dims,
            rff_dim=self.rff_dim,
            kernel_init=init.pytorch_kernel_init,
            bias_init=init.pytorch_bias_init,
        )
        self.terminal = RffReward(
            self.feature_dim,
            self.reward_hidden_dims,
            rff_dim=self.rff_dim,
            kernel_init=init.pytorch_kernel_init,
            bias_init=init.pytorch_bias_init,
        )
        betas, alphas, alphabars = get_noise_schedule("vp", self.num_noises)
        self.alphabars = alphabars

    def forward_phi(self, s, a):
        # s = self.mlp_s(s)
        # a = self.mlp_a(a)
        x = jnp.concat([s, a], axis=-1)
        x = self.mlp_phi(x)
        return x

    def forward_mu(self, sp, t):
        t = self.mlp_t(t)
        x = jnp.concat([sp, t], axis=-1)
        x = self.mlp_mu(x)
        return x.reshape(-1, self.feature_dim, self.obs_dim)

    def forward_reward(self, x: jnp.ndarray):
        return self.reward(x)

    def forward_terminal(self, x: jnp.ndarray):
        return self.terminal(x)

    def __call__(self, rng, s, a, sp, training: bool=False):
        rng, t_rng, eps_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (s.shape[0], 1), 0, self.num_noises+1)
        eps = jax.random.normal(eps_rng, sp.shape)
        spt = jnp.sqrt(self.alphabars[t]) * sp + jnp.sqrt(1-self.alphabars[t]) * eps
        z_phi = self.forward_phi(s, a)
        z_mu = self.forward_mu(spt, t)
        eps_pred = jax.lax.batch_matmul(z_phi[..., jnp.newaxis, :], z_mu)[..., 0, :]
        r_pred = self.forward_reward(z_phi)
        terminal_pred = self.forward_terminal(z_phi)
        return eps, eps_pred, r_pred, terminal_pred, z_phi


@partial(jax.jit, static_argnames=("reward_coef"))
def update_factorized_ddpm(
    rng: PRNGKey,
    ddpm: FactorizedDDPM,
    batch: Batch,
    reward_coef: float,
    terminal_coef: float,
) -> Tuple[PRNGKey, FactorizedDDPM, Metric]:
    B = batch.obs.shape[0]
    rng, update_rng = jax.random.split(rng)
    def loss_fn(ddpm_params: Param, dropout_rng: PRNGKey):
        eps, eps_pred, r_pred, terminal_pred, z_phi = ddpm.apply(
            {"params": ddpm_params},
            update_rng,
            batch.obs,
            batch.action,
            batch.next_obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        ddpm_loss = ((eps_pred - eps) ** 2).sum(axis=-1).mean()
        reward_loss = ((r_pred - batch.reward) ** 2).mean()
        terminal_loss = optax.sigmoid_binary_cross_entropy(terminal_pred, batch.terminal).mean()
        loss = ddpm_loss + reward_coef * reward_loss + terminal_coef * terminal_loss
        return loss, {
            "loss/ddpm_loss": ddpm_loss,
            "loss/reward_loss": reward_loss,
            "loss/terminal_loss": terminal_loss,
            "misc/sp0_mean": batch.next_obs.mean(),
            "misc/sp0_std": batch.next_obs.std(axis=0).mean(),
            "misc/sp0_l1": jnp.abs(batch.next_obs).mean(),
            "misc/eps_mean": eps_pred.mean(),
            "misc/eps_l1": jnp.abs(eps_pred).mean(),
            "misc/reward_mean": r_pred.mean(),
            "misc/z_phi_l1": jnp.abs(z_phi).mean(),
        }

    new_ddpm, metrics = ddpm.apply_gradient(loss_fn)
    return rng, new_ddpm, metrics
