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


@partial(jax.jit, static_argnames=("deterministic", "exploration_noise"))
def jit_sample_action(
    rng: PRNGKey,
    actor: Model,
    obs: jnp.ndarray,
    deterministic: bool,
    exploration_noise: float,
) -> jnp.ndarray:
    action = actor(obs, training=False)
    if not deterministic:
        action = action + exploration_noise * jax.random.normal(rng, action.shape)
        action = jnp.clip(action, -1.0, 1.0)
    return action


def _make_noised(
    sp: jnp.ndarray, num_noises: int, alphabars: jnp.ndarray, rng: jax.random.PRNGKey
):
    if num_noises > 0:
        N = num_noises
        B, D = sp.shape
        sp = jnp.tile(jnp.expand_dims(sp, 0), (num_noises, 1, 1))
        t = jnp.arange(N)
        t = jnp.repeat(t, B).reshape(N,B)
        alpha_t = alphabars[t]  # [N, D, 1]

        sp_exp = jnp.broadcast_to(sp, (N, B, D))
        eps = jax.random.normal(rng, sp_exp.shape) # [N, D, S]

        xt = jnp.sqrt(alpha_t) * sp + jnp.sqrt((1 - alpha_t)) * eps
        t = jnp.expand_dims(t, -1)
    else:
        xt = jnp.expand_dims(sp, 0)
        t = None
    return xt, t


@partial(jax.jit, static_argnames=("num_noises", "linear", "ranking", "reward_coef"))
def update_feature(
    rng: PRNGKey,
    feature: Model,
    batch: Batch,
    alphabars: jnp.ndarray, # "alphabars", "labels" unhashable
    num_noises: int = 0,
    linear: bool = False,  # Fix: feature also has linear?
    ranking: bool = False,
    reward_coef: float = 1.0
) -> Tuple[PRNGKey, Model, Metric]:
    
    # assert batch.obs.shape[0] == cfg.batch_size + cfg.aug_batch_size
    s, a, sp, r = (
        batch.obs,
        batch.action,
        batch.next_obs,
        batch.reward,
    )
    rng, noise_rng = jax.random.split(rng)

    def feature_loss_fn(feature_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        B = s.shape[0]

        z_phi = feature.apply(
            {"params": feature_params}, s, a, method=FactorizedNCE.forward_phi,
            rngs={"dropout": dropout_rng}
        )
        xt, t = _make_noised(sp, num_noises, alphabars, noise_rng)
        z_mu = feature.apply(
            {"params": feature_params}, xt, t, method=FactorizedNCE.forward_mu,
            rngs={"dropout": dropout_rng}
        )

        z_phi = jnp.tile(jnp.expand_dims(z_phi, 0), (z_mu.shape[0], 1, 1))

        logits = jnp.matmul(z_mu, z_phi.transpose(0, 2, 1))
        if linear:
            eff_logits = (softplus_beta(logits, 3.0) + 1e-6).log()
        else:
            eff_logits = logits

        normalizer = feature.apply(
            {"params": feature_params},
            num_noises,
            method=FactorizedNCE.get_normalizer,
            rngs={"dropout": dropout_rng}
        )

        print(f"ranking {ranking}, {eff_logits.shape}, linear: {linear}")

        if ranking:
            labels = jnp.tile(jnp.expand_dims(jnp.arange(B),0), (num_noises, 1))
        else:
            labels = jnp.tile(jnp.expand_dims(jnp.eye(B),0), (num_noises, 1, 1))

        print(f"labels {labels.shape}")

        if linear:
            raise NotImplementedError("linear must be false")
            # eff_logits = (softplus_beta(logits, beta=3.0) + 1e-6).log()
        else:
            eff_logits = logits
            if ranking:
                # We manually broadcast labels here
                labels = jnp.broadcast_to(jnp.expand_dims(labels, -1), eff_logits.shape) # [N, B, B]
                print(f"labels {labels.shape}")
                model_loss = optax.sigmoid_binary_cross_entropy(eff_logits, labels).mean(-1)
            else:
                pass

        print(f"model_loss {model_loss.shape}")

        pred_r = feature.apply(
            {"params": feature_params}, z_phi, method=FactorizedNCE.forward_reward,
            rngs={"dropout": dropout_rng}
        )
        print(f"pred_r {pred_r.shape}")

        model_loss = model_loss.mean()
        reward_loss = jnp.mean((pred_r - r) ** 2)
        total_loss = model_loss + reward_coef * reward_loss

        metrics = {
            "loss/total": total_loss,
            "loss/model_loss": model_loss,
            "loss/reward_loss": reward_loss,
            # "misc/phi_norm": jnp.abs(z_phi).mean().item(),
            # "misc/phi_std": jnp.std(z_phi, 0).mean().item()
        }
        # TODO: add detailed metrics?
        return total_loss, metrics

    new_feature, metrics = feature.apply_gradient(feature_loss_fn)
    return rng, new_feature, metrics


@partial(jax.jit, static_argnames=("discount", "target_policy_noise", "noise_clip"))
def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_target: Model,
    nce_target: Model,
    batch: Batch,
    discount: float,
    target_policy_noise: float,
    noise_clip: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, batch.action.shape) * target_policy_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_action = jnp.clip(actor_target(batch.next_obs) + noise, -1.0, 1.0)

    next_feature = nce_target(batch.next_obs, next_action)
    # q_target = self.critic_target(next_feature).min(0)[0]

    q_target = critic_target(batch.next_obs, next_action).min(axis=0)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target

    def critic_loss_fn(
        critic_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((q - q_target[jnp.newaxis, :]) ** 2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics


@jax.jit
def update_actor(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    batch: Batch,
) -> Tuple[PRNGKey, Model, Metric]:
    def actor_loss_fn(
        actor_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        new_action = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        q = critic(batch.obs, new_action)
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

        self.rng, nce_rng, actor_rng, critic_rng = jax.random.split(self.rng, 4)

        activation = jax.nn.elu # ?
        assert self.aug_batch_size <= self.batch_size, "Aug batch size needs to be lower than batch size"

        _, _, self.alphabars = get_noise_schedule(
            "vp", self.num_noises
        )
        self.alphabars = jnp.expand_dims(self.alphabars, -1)

        actor_def = SquashedDeterministicActor(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.layer_norm,
                dropout=None,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
        )

        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic_hidden_dims,
            layer_norm=cfg.layer_norm,
            activation=activation,
            dropout=None,
            ensemble_size=cfg.critic_ensemble_size,
        )

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
        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.actor_target = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        # hmmm should i make this take in buffer? try with batch first?

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
        critic_batch = Batch(obs, action, reward,terminal, next_obs, None)
        feat_batch = Batch(fobs, faction, freward, fterminal, fnext_obs, None)

        self.rng, self.nce, nce_metrics = update_feature(
            self.rng,
            self.nce,
            feat_batch,
            self.alphabars,
            self.num_noises,
            self.linear,
            self.ranking,
            self.reward_coef
        )
        metrics.update(nce_metrics)

        self.rng, self.critic, critic_metrics = update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor_target,
            self.nce_target,
            critic_batch,
            discount=self.cfg.discount,
            target_policy_noise=self.target_policy_noise,
            noise_clip=self.noise_clip,
        )
        metrics.update(critic_metrics)

        if self._n_training_steps % self.actor_update_freq == 0:
            self.rng, self.actor, actor_metrics = update_actor(
                self.rng,
                self.actor,
                self.critic,
                critic_batch,
            )
            metrics.update(actor_metrics)

        if self._n_training_steps % self.target_update_freq == 0:
            self.critic_target = ema_update(
                self.critic, self.critic_target, self.cfg.ema
            )
            self.actor_target = ema_update(self.actor, self.actor_target, self.cfg.ema)
            self.nce_target = ema_update(self.nce, self.nce_target, self.feature_tau)

        self._n_training_steps += 1
        return metrics
