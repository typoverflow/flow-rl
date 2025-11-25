from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.flow.continuous_ddpm import cosine_noise_schedule
from flowrl.flow.ddpm import get_noise_schedule
from flowrl.functional.activation import l2_normalize, mish
from flowrl.module.critic import Critic
from flowrl.module.mlp import MLP, ResidualMLP
from flowrl.module.model import Model
from flowrl.module.rff import RffReward
from flowrl.module.time_embedding import LearnableFourierEmbedding, PositionalEmbedding
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
        x = jnp.concatenate([item for item in [ft, t_ff] if item is not None], axis=-1)
        x = ResidualMLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            multiplier=1,
            activation=self.activation,
            layer_norm=True,
        )(x, training)
        return x

class EnsembleResidualCritic(nn.Module):
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    ensemble_size: int = 2

    @nn.compact
    def __call__(
        self,
        obs: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        vmap_critic = nn.vmap(
            ResidualCritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        x = vmap_critic(
            time_embedding=self.time_embedding,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
        )(obs, action, t, training)
        return x

class ACACritic(nn.Module):
    time_dim: int
    hidden_dims: Sequence[int]
    activation: Callable

    @nn.compact
    def __call__(self, obs, action, t, training=False):
        t_ff = PositionalEmbedding(self.time_dim)(t)
        t_ff = nn.Dense(2*self.time_dim)(t_ff)
        t_ff = self.activation(t_ff)
        t_ff = nn.Dense(self.time_dim)(t_ff)
        x = jnp.concatenate([obs, action, t_ff], axis=-1)
        return MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            layer_norm=False
        )(x, training)

class EnsembleACACritic(nn.Module):
    time_dim: int
    hidden_dims: Sequence[int]
    activation: Callable
    ensemble_size: int

    @nn.compact
    def __call__(self, obs, action, t, training=False):
        vmap_critic = nn.vmap(
            ACACritic,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        return vmap_critic(
            time_dim=self.time_dim,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
        )(obs, action, t, training)


class SeparateCritic(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable
    ensemble_size: int

    @nn.compact
    def __call__(self, obs, action, t, training=False):
        vmap_critic = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=-1,
            axis_size=self.ensemble_size
        )
        x = jnp.concatenate([obs, action], axis=-1)
        out = vmap_critic(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            layer_norm=False
        )(x, training)
        out = out.reshape(*out.shape[:-2], -1)
        # Using jnp.take_along_axis for batched index selection (broadcasting as needed)
        # out: (E, B, T, 1), need to select on axis=2 using t_indices[b] for each batch
        # We assume batch dim is axis=1, time axis=2
        out = jnp.take_along_axis(
            out,
            t.astype(jnp.int32),
            axis=-1
        )
        return out

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
        self.mlp_t1 = nn.Sequential(
            [PositionalEmbedding(128), nn.Dense(256), mish, nn.Dense(128)]
        )
        self.mlp_t2 = nn.Sequential(
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
            [512,],
            rff_dim=self.rff_dim,
        )
        # self.reward = Critic(
        #     hidden_dims=self.reward_hidden_dims,
        #     activation=nn.elu,
        #     layer_norm=True,
        #     dropout=None,
        # )
        if self.num_noises > 0:
            self.use_noise_perturbation = True
            from flowrl.flow.ddpm import cosine_beta_schedule
            betas = cosine_beta_schedule(T=self.num_noises)
            betas = jnp.concatenate([jnp.zeros((1,)), betas])
            alphas = 1 - betas
            self.alpha_hats = jnp.cumprod(alphas)
        else:
            self.use_noise_perturbation = False
        self.N = max(self.num_noises, 1)
        if not self.ranking:
            self.normalizer = self.param("normalizer", lambda key: jnp.zeros((self.N,), jnp.float32))
        else:
            self.normalizer = self.param("normalizer", lambda key: jnp.zeros((self.N,), jnp.float32))

    def forward_phi(self, s, at, t):
        x = jnp.concat([s, at], axis=-1)
        t_ff = self.mlp_t1(t)
        x = jnp.concat([x, t_ff], axis=-1)
        x = self.mlp_phi(x)
        x = l2_normalize(x, group_size=None)
        # x = jnp.tanh(x)
        return x

    def forward_mu(self, sp, t):
        t_ff = self.mlp_t2(t)
        sp = jnp.concat([sp, t_ff], axis=-1)
        sp = self.mlp_mu(sp)
        sp = jnp.tanh(sp)
        return sp

    def forward_reward(self, x: jnp.ndarray):  # for z_phi
        return self.reward(x)

    def forward_logits(
        self,
        rng: PRNGKey,
        s: jnp.ndarray,
        a: jnp.ndarray,
        sp: jnp.ndarray,
    ):
        B, D = sp.shape
        rng, t_rng, eps1_rng, eps2_rng = jax.random.split(rng, 4)
        if self.use_noise_perturbation:

            # perturb sp, (N, B, D) with the noise level shared in each N
            sp0 = jnp.broadcast_to(sp, (self.N, B, sp.shape[-1]))
            t_sp = jnp.arange(1, self.num_noises+1)
            t_sp = jnp.repeat(t_sp, B).reshape(self.N, B, 1)
            eps_sp = jax.random.normal(eps1_rng, sp0.shape)
            alpha, sigma = jnp.sqrt(self.alpha_hats[t_sp]), jnp.sqrt(1 - self.alpha_hats[t_sp])
            spt = alpha * sp0 + sigma * eps_sp

            # perturb a, (N, B, D) with noise level independent across N
            a0 = a
            t_a = jax.random.randint(t_rng, (B, 1), 1, self.num_noises+1)
            # t_a = jnp.ones((B, 1), dtype=jnp.int32)
            eps_a = jax.random.normal(eps2_rng, a0.shape)
            alpha, sigma = jnp.sqrt(self.alpha_hats[t_a]), jnp.sqrt(1 - self.alpha_hats[t_a])
            at = alpha * a0 + sigma * eps_a
            # at = jnp.broadcast_to(at, (self.N, B, at.shape[-1]))
            # t_a = jnp.broadcast_to(t_a, (self.N, B, 1))
        else:
            s = jnp.expand_dims(s, 0)
            at = jnp.expand_dims(a, 0)
            t = None
        z_phi = self.forward_phi(s, at, t_a)
        z_mu = self.forward_mu(spt, t_sp)
        logits = jax.lax.batch_matmul(
            jnp.broadcast_to(z_phi, (self.N, B, z_phi.shape[-1])),
            jnp.swapaxes(z_mu, -1, -2)
        )
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
        logits, rewards = nce.apply(
            {"params": nce_params},
            logits_rng,
            batch.obs,
            batch.action,
            batch.next_obs,
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
