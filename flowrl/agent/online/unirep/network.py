from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.flow.continuous_ddpm import cosine_noise_schedule
from flowrl.flow.ddpm import get_noise_schedule
from flowrl.functional.activation import l2_normalize, mish
from flowrl.module.critic import Critic
from flowrl.module.mlp import ResidualMLP
from flowrl.module.model import Model
from flowrl.module.rff import RffReward
from flowrl.module.time_embedding import LearnableFourierEmbedding
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
            [LearnableFourierEmbedding(128), nn.Dense(256), mish, nn.Dense(128)]
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
        # self.reward = RffReward(
        #     self.feature_dim,
        #     self.reward_hidden_dims,
        #     rff_dim=self.rff_dim,
        # )
        self.reward = Critic(
            hidden_dims=self.reward_hidden_dims,
            activation=nn.elu,
            layer_norm=True,
            dropout=None,
        )
        if self.num_noises > 0:
            self.use_noise_perturbation = True
            self.noise_schedule_fn = cosine_noise_schedule
        else:
            self.use_noise_perturbation = False
        self.N = max(self.num_noises, 1)
        if not self.ranking:
            self.normalizer = self.param("normalizer", lambda key: jnp.zeros((self.N,), jnp.float32))
        else:
            self.normalizer = self.param("normalizer", lambda key: jnp.zeros((self.N,), jnp.float32))

    def forward_phi(self, s, at, t):
        x = jnp.concat([s, at], axis=-1)
        if t is not None:
            t_ff = self.mlp_t(t)
            x = jnp.concat([x, t_ff], axis=-1)
        x = self.mlp_phi(x)
        x = l2_normalize(x, group_size=None)
        return x

    def forward_mu(self, sp):
        sp = self.mlp_mu(sp)
        return sp

    def forward_reward(self, x: jnp.ndarray):  # for z_phi
        return self.reward(x)

    def forward_logits(
        self,
        rng: PRNGKey,
        s: jnp.ndarray,
        a: jnp.ndarray,
        sp: jnp.ndarray,
        z_mu: jnp.ndarray | None=None
    ):
        B, D = sp.shape
        rng, t_rng, eps_rng = jax.random.split(rng, 3)
        if z_mu is None:
            z_mu = self.forward_mu(sp)
        if self.use_noise_perturbation:
            s = jnp.broadcast_to(s, (self.N, B, s.shape[-1]))
            a0 = jnp.broadcast_to(a, (self.N, B, a.shape[-1]))
            t = jax.random.uniform(t_rng, (self.N,), dtype=jnp.float32) # check removing min val and max val is valid
            t = jnp.repeat(t, B).reshape(self.N, B, 1)
            eps = jax.random.normal(eps_rng, a0.shape)
            alpha, sigma = self.noise_schedule_fn(t)
            at = alpha * a0 + sigma * eps
        else:
            s = jnp.expand_dims(s, 0)
            at = jnp.expand_dims(a, 0)
            t = None
        z_phi = self.forward_phi(s, at, t)
        z_mu = jnp.broadcast_to(z_mu, (self.N, B, self.feature_dim))
        logits = jax.lax.batch_matmul(z_phi, jnp.swapaxes(z_mu, -1, -2))
        logits = logits / jnp.exp(self.normalizer[:, None, None])
        rewards = self.forward_reward(z_phi)
        return logits, rewards

    def forward_normalizer(self):
        return self.normalizer

    def __call__(
        self,
        rng: PRNGKey,
        s,
        a,
        sp,
    ):
        logits, rewards = self.forward_logits(rng, s, a, sp)
        _ = self.forward_normalizer()

        return logits, rewards


@partial(jax.jit, static_argnames=("ranking", "reward_coef"))
def update_factorized_nce(
    rng: PRNGKey,
    nce: Model,
    batch: Batch,
    ranking: bool,
    reward_coef: float,
) -> Tuple[PRNGKey, Model, Metric]:
    B = batch.obs.shape[0]
    rng, logits_rng = jax.random.split(rng)
    if ranking:
        labels = jnp.arange(B)
    else:
        labels = jnp.eye(B)

    def loss_fn(nce_params: Param, dropout_rng: PRNGKey):
        z_mu = nce.apply({"params": nce_params}, batch.next_obs, method="forward_mu")
        logits, rewards = nce.apply(
            {"params": nce_params},
            logits_rng,
            batch.obs,
            batch.action,
            batch.next_obs,
            z_mu,
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
        normalizer = nce.apply({"params": nce_params}, method="forward_normalizer")
        rewards_target = jnp.broadcast_to(batch.reward, rewards.shape)
        reward_loss = jnp.mean((rewards - rewards_target) ** 2)

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
            "misc/obs_mean": batch.obs.mean(),
            "misc/obs_std": batch.obs.std(axis=0).mean(),
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
