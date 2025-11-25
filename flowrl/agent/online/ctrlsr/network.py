from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

import flowrl.module.initialization as init
from flowrl.flow.ddpm import get_noise_schedule
from flowrl.functional.activation import l2_normalize, mish
from flowrl.module.mlp import MLP, ResidualMLP
from flowrl.module.model import Model
from flowrl.module.rff import RffReward
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import *
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
    ranking: bool = False

    def setup(self):
        MLP_torch_init = partial(MLP, kernel_init=init.pytorch_kernel_init, bias_init=init.pytorch_bias_init)
        self.mlp_t = nn.Sequential([
            PositionalEmbedding(128),
            MLP_torch_init(output_dim=256),
            mish,
            MLP_torch_init(output_dim=128)
        ])
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
            self.feature_dim,
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
        if self.num_noises > 0:
            self.use_noise_perturbation = True
            betas, alphas, alphabars = get_noise_schedule("vp", self.num_noises)
            self.alphabars = alphabars
        else:
            self.use_noise_perturbation = False
        self.N = max(self.num_noises, 1)

    def forward_phi(self, s, a):
        x = jnp.concat([s, a], axis=-1)
        x = self.mlp_phi(x)
        return x

    def forward_mu(self, sp, t=None):
        if t is not None:
            t_ff = self.mlp_t(t)
            sp = jnp.concat([sp, t_ff], axis=-1)
        sp = self.mlp_mu(sp)
        sp = jnp.tanh(sp)
        return sp

    def forward_reward(self, x: jnp.ndarray):  # for z_phi
        return self.reward(x)

    def __call__(self, rng, s, a, sp, training: bool=False):
        B, D = sp.shape
        rng, eps_rng = jax.random.split(rng, 2)
        z_phi = self.forward_phi(s, a)
        if self.use_noise_perturbation:
            sp = jnp.broadcast_to(sp, (self.N, B, D))
            t = jnp.arange(self.num_noises)
            t = jnp.repeat(t, B).reshape(self.N, B, 1)
            alphabars = self.alphabars[t]
            eps = jax.random.normal(eps_rng, sp.shape)
            xt = jnp.sqrt(alphabars) * sp + jnp.sqrt(1-alphabars) * eps
        else:
            xt = jnp.expand_dims(sp, 0)
            t = None
        z_mu = self.forward_mu(xt, t)
        logits = jax.lax.batch_matmul(
            jnp.broadcast_to(z_phi, (self.N, B, self.feature_dim)),
            jnp.swapaxes(z_mu, -1, -2)
        )
        r_pred = self.forward_reward(z_phi)
        return logits, r_pred, z_phi


@partial(jax.jit, static_argnames=("ranking", "reward_coef"))
def update_factorized_nce(
    rng: PRNGKey,
    nce: Model,
    batch: Batch,
    ranking: bool,
    reward_coef: float,
) -> Tuple[PRNGKey, Model, Metric]:
    B = batch.obs.shape[0]
    rng, update_rng = jax.random.split(rng)
    if ranking:
        labels = jnp.arange(B)
    else:
        labels = jnp.eye(B)

    def loss_fn(nce_params: Param, dropout_rng: PRNGKey):
        logits, r_pred, z_phi = nce.apply(
            {"params": nce_params},
            update_rng,
            batch.obs,
            batch.action,
            batch.next_obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        if ranking:
            model_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, jnp.broadcast_to(labels, (logits.shape[0], B))
            ).mean(axis=-1)
        else:
            raise NotImplementedError("non-ranking mode is not supported")
        reward_loss = jnp.mean((r_pred - batch.reward) ** 2)

        nce_loss = model_loss.mean() + reward_coef * reward_loss

        pos_logits = logits[
            jnp.arange(logits.shape[0])[..., jnp.newaxis],
            jnp.arange(logits.shape[1]),
            jnp.arange(logits.shape[2])[jnp.newaxis, ...].repeat(logits.shape[0], axis=0)
        ]
        pos_logits_per_noise = pos_logits.mean(axis=-1)
        neg_logits = (logits.sum(axis=-1) - pos_logits) / (logits.shape[-1] - 1)
        neg_logits_per_noise = neg_logits.mean(axis=-1)
        metrics = {
            "loss/nce_loss": nce_loss,
            "loss/model_loss": model_loss.mean(),
            "loss/reward_loss": reward_loss,
            "misc/obs_mean": batch.obs.mean(),
            "misc/obs_std": batch.obs.std(axis=0).mean(),
            "misc/phi_l1": jnp.abs(z_phi).mean(),
        }
        checkpoints = list(range(0, logits.shape[0], logits.shape[0]//5)) + [logits.shape[0]-1]
        metrics.update({
            f"misc/positive_logits_{i}": pos_logits_per_noise[i].mean() for i in checkpoints
        })
        metrics.update({
            f"misc/negative_logits_{i}": neg_logits_per_noise[i].mean() for i in checkpoints
        })
        metrics.update({
            f"misc/logits_gap_{i}": (pos_logits_per_noise[i] - neg_logits_per_noise[i]).mean() for i in checkpoints
        })
        return nce_loss, metrics

    new_nce, metrics = nce.apply_gradient(loss_fn)
    return rng, new_nce, metrics
