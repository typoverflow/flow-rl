from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from operator import attrgetter

from flowrl.agent.base import BaseAgent
from flowrl.config.online.mujoco.algo.ctrl import CTRL_TD3_Config
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedDeterministicActor
from flowrl.module.critic import EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey
from flowrl.agent.online.ctrl.network import FactorizedNCE
from flowrl.flow.ddpm import get_noise_schedule
from flowrl.functional.activation import softplus_beta
from flowrl.agent.online.td3 import TD3Agent
from flowrl.module.rff import RffDoubleQ


def compute_logits(
    feature: Model,
    feature_params: Param,
    s: jnp.ndarray,
    a: jnp.ndarray,
    sp: jnp.ndarray,
    num_noises: int,
    alphabars: jnp.ndarray,
    noise_rng: jax.random.PRNGKey,
    mu_dropout_rng: jax.random.PRNGKey,
    z_phi: jnp.ndarray | None = None,
):
    B, D = sp.shape
    if z_phi is None:
        z_phi = feature.apply(
            {"params": feature_params}, s, a, method=FactorizedNCE.forward_phi
        )

    if num_noises > 0:
        sp_exp = jnp.broadcast_to(sp, (num_noises, B, D))
        t = jnp.arange(num_noises)
        t = jnp.repeat(t, B).reshape(num_noises, B)
        alpha_t = alphabars[t]  # [N, D, 1]
        eps = jax.random.normal(noise_rng, sp_exp.shape)  # [N, D, S]
        xt = jnp.sqrt(alpha_t) * sp_exp + jnp.sqrt((1 - alpha_t)) * eps
        t = jnp.expand_dims(t, -1)
    else:
        xt = jnp.expand_dims(sp, 0)
        t = None

    z_mu = feature.apply(
        {"params": feature_params},
        xt,
        t,
        method=FactorizedNCE.forward_mu,
        rngs={"dropout": mu_dropout_rng},
    )  # (N, B, D)

    logits = jnp.matmul(z_mu, jnp.swapaxes(z_phi, -1, -2))
    return logits


@partial(jax.jit, static_argnames=("num_noises", "linear", "ranking", "reward_coef"))
def update_feature(
    rng: PRNGKey,
    feature: Model,
    batch: Batch,
    alphabars: jnp.ndarray,
    num_noises: int = 0,
    linear: bool = False,
    ranking: bool = False,
    reward_coef: float = 1.0,
) -> Tuple[PRNGKey, Model, Metric]:
    s, a, sp, r = (
        batch.obs,
        batch.action,
        batch.next_obs,
        batch.reward,
    )

    def feature_loss_fn(
        feature_params: Param, rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        B = s.shape[0]
        rng, rng_phi, rng_mu, rng_normalizer, rng_r, noise_rng = jax.random.split(
            rng, 6
        )

        z_phi = feature.apply(
            {"params": feature_params},
            s,
            a,
            method=FactorizedNCE.forward_phi,
            rngs={"dropout": rng_phi},
        )  # (B, D)
        logits = compute_logits(
            feature,
            feature_params,
            s,
            a,
            sp,
            num_noises,
            alphabars,
            noise_rng,
            rng_mu,
            z_phi,
        )

        normalizer = feature.apply(
            {"params": feature_params},
            method=FactorizedNCE.get_normalizer,
            rngs={"dropout": rng_normalizer},
        )

        if ranking:
            labels = jnp.tile(
                jnp.expand_dims(jnp.arange(B, dtype=jnp.int32), 0), (num_noises, 1)
            )
        else:
            labels = jnp.tile(
                jnp.expand_dims(jnp.eye(B, dtype=logits.dtype), 0), (num_noises, 1, 1)
            )

        if linear:
            eff_logits = jnp.log(softplus_beta(logits, 3.0) + 1e-6)
            if ranking:
                model_loss = optax.softmax_cross_entropy_with_integer_labels(
                    eff_logits, labels
                ).mean(-1)
            else:
                eff_logits = eff_logits * jnp.exp(normalizer)[:, None, None] / B
                model_loss = optax.sigmoid_binary_cross_entropy(
                    eff_logits, labels
                ).mean([-2, -1])
        else:
            eff_logits = logits
            if ranking:
                model_loss = optax.softmax_cross_entropy_with_integer_labels(
                    eff_logits, labels
                ).mean(-1)
            else:
                eff_logits = eff_logits + normalizer[:, None, None] - jnp.log(B)
                model_loss = optax.sigmoid_binary_cross_entropy(
                    eff_logits, labels
                ).mean([-2, -1])

        pred_r = feature.apply(
            {"params": feature_params},
            z_phi,
            method=FactorizedNCE.forward_reward,
            rngs={"dropout": rng_r},
        )

        model_loss = model_loss.mean()
        reward_loss = jnp.mean((pred_r.squeeze(-1) - r) ** 2)
        total_loss = model_loss + reward_coef * reward_loss

        metrics = {
            "loss/total": total_loss,
            "loss/model_loss": model_loss,
            "loss/reward_loss": reward_loss,
        }
        return total_loss, metrics

    new_feature, metrics = feature.apply_gradient(feature_loss_fn)
    return rng, new_feature, metrics


@partial(jax.jit, static_argnames=("discount", "target_policy_noise", "noise_clip"))
def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_target: Model,
    feature: Model,
    feature_target: Model,
    batch: Batch,
    discount: float,
    target_policy_noise: float,
    noise_clip: float,
    critic_coef: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    noise = jnp.clip(
        jax.random.normal(sample_rng, batch.action.shape) * target_policy_noise,
        -noise_clip,
        noise_clip,
    )
    next_action = jnp.clip(actor_target(batch.next_obs) + noise, -1.0, 1.0)

    rng, dropout_rng, dropout_rng2 = jax.random.split(rng, 3)
    # need dropout rng here right?
    next_feature = feature_target.apply(
        {"params": feature_target.params},
        batch.next_obs,
        next_action,
        method=FactorizedNCE.forward_phi,
        rngs={"dropout": dropout_rng},
    )

    q_target = critic_target(next_feature).min(0)  # take min of two nets
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target

    back_critic_grad = False
    if back_critic_grad:
        raise NotImplementedError("no back critic grad exists")
    else:
        cur_feature = feature_target.apply(
            {"params": feature_target.params},
            batch.obs,
            batch.action,
            method=FactorizedNCE.forward_phi,
            rngs={"dropout": dropout_rng2},
        )

    def critic_loss_fn(
        critic_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        q_pred = critic.apply(
            {"params": critic_params},
            cur_feature,
            rngs={"dropout": dropout_rng},
        )
        # TODO: why do i need to reshape?
        critic_loss = (
            critic_coef * ((q_pred - q_target[jnp.newaxis, :]) ** 2).sum(0).mean()
        )
        return critic_loss, {
            "loss/critic_loss": critic_loss,
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics


@jax.jit
def update_actor(
    rng: PRNGKey,
    actor: Model,
    feature_target: Model,
    critic: Model,
    batch: Batch,
) -> Tuple[PRNGKey, Model, Metric]:
    def actor_loss_fn(
        actor_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        dropout_rng1, dropout_rng2 = jax.random.split(dropout_rng)
        new_action = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng1},
        )

        # TODO: back critic grad?
        # TODO: hmmm not sure dropout_rng2 right, should do eval?
        new_feature = feature_target.apply(
            {"params": feature_target.params},
            batch.obs,
            new_action,
            method=FactorizedNCE.forward_phi,
            rngs={"dropout": dropout_rng2},
        )
        q = critic(new_feature)
        actor_loss = -q.mean()

        return actor_loss, {
            "loss/actor_loss": actor_loss,
        }

    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, metrics


class Ctrl_TD3_Agent(TD3Agent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    """

    name = "CTRLTD3Agent"
    model_names = ["nce", "actor", "actor_target", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: CTRL_TD3_Config, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.actor_update_freq = cfg.actor_update_freq
        self.target_update_freq = cfg.target_update_freq
        self.target_policy_noise = cfg.target_policy_noise
        self.noise_clip = cfg.noise_clip
        self.exploration_noise = cfg.exploration_noise

        self.ctrl_coef = cfg.ctrl_coef
        self.critic_coef = cfg.critic_coef

        self.batch_size = cfg.batch_size
        self.aug_batch_size = cfg.aug_batch_size
        self.feature_tau = cfg.feature_tau
        self.linear = cfg.linear
        self.ranking = cfg.ranking
        self.feature_dim = cfg.feature_dim
        self.num_noises = cfg.num_noises
        self.reward_coef = cfg.reward_coef
        self.rff_dim = cfg.rff_dim

        self.rng, nce_rng, actor_rng, critic_rng = jax.random.split(self.rng, 4)

        assert self.aug_batch_size <= self.batch_size, (
            "Aug batch size needs to be lower than batch size"
        )

        _, _, self.alphabars = get_noise_schedule("vp", self.num_noises)
        self.alphabars = jnp.expand_dims(self.alphabars, -1)

        nce_def = FactorizedNCE(
            self.obs_dim,
            self.act_dim,
            self.feature_dim,
            cfg.phi_hidden_dims,
            cfg.mu_hidden_dims,
            cfg.reward_hidden_dims,
            cfg.rff_dim,
            cfg.num_noises,
            self.linear,
            self.ranking,
        )
        self.nce = Model.create(
            nce_def,
            nce_rng,
            inputs=(
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
            optimizer=optax.adam(learning_rate=cfg.feature_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.nce_target = Model.create(
            nce_def,
            nce_rng,
            inputs=(
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
        )

        actor_def = SquashedDeterministicActor(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.layer_norm,
                dropout=None,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
        )
        critic_def = RffDoubleQ(
            feature_dim=self.feature_dim,
            hidden_dims=cfg.critic_hidden_dims,
            linear=cfg.linear,
            rff_dim=cfg.rff_dim,
        )
        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.actor_target = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
        )

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        metrics = {}

        split_index = self.batch_size - self.aug_batch_size
        obs, action, next_obs, reward, terminal = [
            b[:split_index]
            for b in attrgetter("obs", "action", "next_obs", "reward", "terminal")(
                batch
            )
        ]
        fobs, faction, fnext_obs, freward, fterminal = [
            b[split_index:]
            for b in attrgetter("obs", "action", "next_obs", "reward", "terminal")(
                batch
            )
        ]
        critic_batch = Batch(obs, action, reward, terminal, next_obs, None)
        feat_batch = Batch(fobs, faction, freward, fterminal, fnext_obs, None)

        self.rng, self.nce, nce_metrics = update_feature(
            self.rng,
            self.nce,
            feat_batch,
            self.alphabars,
            self.num_noises,
            self.linear,
            self.ranking,
            self.reward_coef,
        )
        metrics.update(nce_metrics)

        self.rng, self.critic, critic_metrics = update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor_target,
            self.nce,
            self.nce_target,
            critic_batch,
            discount=self.cfg.discount,
            target_policy_noise=self.target_policy_noise,
            noise_clip=self.noise_clip,
            critic_coef=self.critic_coef,
        )
        metrics.update(critic_metrics)

        if self._n_training_steps % self.actor_update_freq == 0:
            self.rng, self.actor, actor_metrics = update_actor(
                self.rng,
                self.actor,
                self.nce_target,
                self.critic,
                critic_batch,
            )
            metrics.update(actor_metrics)

        if self._n_training_steps % self.target_update_freq == 0:
            self.sync_target()

        self._n_training_steps += 1
        return metrics

    def sync_target(self):
        super().sync_target()
        self.nce_target = ema_update(self.nce, self.nce_target, self.feature_tau)
