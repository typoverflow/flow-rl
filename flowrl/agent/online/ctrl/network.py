from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.flow.ddpm import get_noise_schedule
from flowrl.functional.activation import group_l2_normalize, mish
from flowrl.module.mlp import ResidualMLP
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
        self.mlp_t = nn.Sequential(
            [PositionalEmbedding(128), nn.Dense(256), mish, nn.Dense(128)]
        )
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
            self.feature_dim,
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
        if self.num_noises > 0:
            self.use_noise_perturbation = True
            betas, alphas, alphabars = get_noise_schedule("vp", self.num_noises)
            alphabars_prev = jnp.pad(alphabars[:-1], (1, 0), constant_values=1.0)
            self.betas = betas[..., jnp.newaxis]
            self.alphas = alphas[..., jnp.newaxis]
            self.alphabars = alphabars[..., jnp.newaxis]
            self.alphabars_prev = alphabars_prev[..., jnp.newaxis]
        else:
            self.use_noise_perturbation = False
        self.N = max(self.num_noises, 1)
        if not self.ranking:
            self.normalizer = self.param("normalizer", lambda key: jnp.zeros((self.N,), jnp.float32))
        else:
            self.normalizer = self.param("normalizer", lambda key: jnp.zeros((self.N,), jnp.float32))

    def forward_phi(self, s, a):
        x = jnp.concat([s, a], axis=-1)
        x = self.mlp_phi(x)
        x = group_l2_normalize(x, group_size=None)
        return x

    def forward_mu(self, sp, t=None):
        if t is not None:
            t_ff = self.mlp_t(t)
            sp = jnp.concat([sp, t_ff], axis=-1)
        sp = self.mlp_mu(sp)
        # return jnp.tanh(sp)
        return sp

    def forward_reward(self, x: jnp.ndarray):  # for z_phi
        return self.reward(x)

    def forward_logits(
        self,
        rng: PRNGKey,
        s: jnp.ndarray,
        a: jnp.ndarray,
        sp: jnp.ndarray,
        z_phi: jnp.ndarray | None=None
    ):
        B, D = sp.shape
        rng, eps_rng = jax.random.split(rng, 2)
        if z_phi is None:
            z_phi = self.forward_phi(s, a)
        if self.use_noise_perturbation:
            sp = jnp.broadcast_to(sp, (self.N, B, D))
            t = jnp.arange(self.num_noises)
            t = jnp.repeat(t, B).reshape(self.N, B)
            alphabars = self.alphabars[t]
            eps = jax.random.normal(eps_rng, sp.shape)
            xt = jnp.sqrt(alphabars) * sp + jnp.sqrt(1-alphabars) * eps
            t = jnp.expand_dims(t, -1)
        else:
            xt = jnp.expand_dims(sp, 0)
            t = None
        z_mu = self.forward_mu(xt, t)
        z_phi = jnp.broadcast_to(z_phi, (self.N, B, self.feature_dim))
        logits = jax.lax.batch_matmul(z_phi, jnp.swapaxes(z_mu, -1, -2))
        logits = logits / jnp.exp(self.normalizer[:, None, None])
        return logits

    def forward_normalizer(self):
        return self.normalizer

    def __call__(
        self,
        rng: PRNGKey,
        s,
        a,
        sp,
    ):
        z_phi = self.forward_phi(s, a)
        _ = self.forward_reward(z_phi)
        _ = self.forward_logits(rng, s, a, sp, z_phi=z_phi)

        _ = self.forward_normalizer()

        return z_phi


@partial(jax.jit, static_argnames=("ranking", "reward_coef"))
def update_factorized_nce(
    rng: PRNGKey,
    nce: Model,
    s: jnp.ndarray,
    a: jnp.ndarray,
    sp: jnp.ndarray,
    r: jnp.ndarray,
    ranking: bool,
    reward_coef: float,
) -> Tuple[PRNGKey, Model, Metric]:
    B = s.shape[0]
    rng, logits_rng = jax.random.split(rng)
    if ranking:
        labels = jnp.arange(B)
    else:
        labels = jnp.eye(B)

    def loss_fn(nce_params: Param, dropout_rng: PRNGKey):
        z_phi = nce.apply(
            {"params": nce_params},
            s,
            a,
            method="forward_phi",
        )
        logits = nce.apply(
            {"params": nce_params},
            logits_rng,
            s,
            a,
            sp,
            z_phi,
            method="forward_logits",
        )

        if ranking:
            model_loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, jnp.broadcast_to(labels, (logits.shape[0], B))
            ).mean(axis=-1)
        else:
            normalizer = nce.apply({"params": nce_params}, method="forward_normalizer")
            eff_logits = logits + normalizer[:, None, None] - jnp.log(B)
            model_loss = optax.sigmoid_binary_cross_entropy(eff_logits, labels).mean([-2, -1])
        r_pred = nce.apply(
            {"params": nce_params},
            z_phi,
            method="forward_reward",
        )
        normalizer = nce.apply({"params": nce_params}, method="forward_normalizer")
        reward_loss = jnp.mean((r_pred - r) ** 2)

        nce_loss = model_loss.mean() + reward_coef * reward_loss + 0.000 * (logits**2).mean()

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
            "misc/obs_mean": s.mean(),
            "misc/obs_std": s.std(axis=0).mean(),
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
        metrics.update({
            f"misc/normalizer_{i}": jnp.exp(normalizer[i]) for i in checkpoints
        })
        return nce_loss, metrics

    new_nce, metrics = nce.apply_gradient(loss_fn)
    return rng, new_nce, metrics
