from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.mujoco.algo.td3 import TD3Config
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedDeterministicActor
from flowrl.module.critic import EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import Batch, Metric, Param, PRNGKey


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


# TODO: update_feature
@partial(jax.jit, static_argnames=("num_noises", "linear", "ranking"))
def update_feature(
    rng: PRNGKey,
    feature: Model,
    # normalizer is an optional model?
    batch: Batch,
    num_noises: int = 0,
    linear: bool = False,  # Fix: feature also has linear?
    ranking: bool = False,
) -> Tuple[PRNGKey, Model, Metric]:
    B = batch.next_obs.shape[0]
    rng, noise_rng = jax.random.split(rng)
    use_noise_perturbation = True if num_noises > 0 else False

    s, a, sp, r, terminal = (
        batch.obs,
        batch.action,
        batch.next_obs,
        batch.reward,
        batch.terminal,
    )

    # TODO: get alphas, betas

    if use_noise_perturbation:
        # this is not right!
        sp = jnp.expand_dims(sp, 0).repeat([num_noises, 1, 1])
        t = jnp.arange(0, num_noises)
        # TODO: fix here
        t = jnp.repeat_interleave
        alphabars = self.alphabars[t]
        eps = jax.random.normal(noise_rng, sp.shape)
        xt = jnp.sqrt(alphabars) * sp + jnp.sqrt((1 - alphabars)) * eps
        t = jnp.expand_dims(t, -1)
    else:
        xt = jnp.expand_dims(sp, 0)
        t = None

    def compute_logits(self, s, a, sp, z_phi=None):
        if z_phi is None:
            z_phi = self.forward_phi(s, a)
        z_mu = self.forward_mu(xt, t)  # (N, RB, z_dim)
        z_phi = jnp.expand_dims(z_phi, 0).repeat(z_mu.shape[0], 1, 1)  # (N, LB, z_dim)
        # TODO: has a batch matmul?
        logits = jnp.matmul(z_mu, z_phi)  # (N, LB, RB)
        return logits

    def feature_loss_fn(feature_params: Param) -> Tuple[jnp.ndarray, Metric]:
        B = s.shape[0]
        N = 1 if num_noises <= 0 else num_noises
        # z_phi = feature.forward_phi(s, a)
        z_phi = feature.apply(s, a, method=feature.forward_phi)
        logits = self.compute_logits(s, a, sp, z_phi)

        if linear:
            # TODO: wrong
            eff_logits = jnp.log(softplus_beta(logits, beta=3.0) + 1e-6)
            # if self.ranking:
            #     model_loss =

        batch_size = batch.obs.shape[0]  # ???
        if ranking:
            labels = jnp.expand_dims(jnp.arange(batch_size), axis=0).repeat(
                num_noises, 1
            )
        else:
            # repeats, is this right?
            labels = jnp.expand_dims(jnp.eye(batch_size), axis=0).repeat(
                [num_noises, 1, 1]
            )
            # self.normalizer = self.variable(
            #     "normalizer",
            #     "etc",
            #     jnp.zeros(
            #         [max(self.num_noises, 1)], dtype=jnp.float32, device=self.device
            #     ),
            # )
        reward_loss = MSE()

        return loss, metrics

    new_feature, metrics = feature.apply_gradient(feature_loss_fn)


@partial(jax.jit, static_argnames=("discount", "target_policy_noise", "noise_clip"))
def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_target: Model,
    batch: Batch,
    discount: float,
    target_policy_noise: float,
    noise_clip: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, batch.action.shape) * target_policy_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_action = jnp.clip(actor_target(batch.next_obs) + noise, -1.0, 1.0)

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


class Ctrl_TD3_Agent(BaseAgent):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    """

    name = "CTRLTD3Agent"
    model_names = ["nce", "actor", "actor_target", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: TD3Config, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.actor_update_freq = cfg.actor_update_freq
        self.target_update_freq = cfg.target_update_freq
        self.target_policy_noise = cfg.target_policy_noise
        self.noise_clip = cfg.noise_clip
        self.exploration_noise = cfg.exploration_noise

        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        activation = {
            "relu": jax.nn.relu,
            "elu": jax.nn.elu,
        }[cfg.activation]

        # TODO: also we are syncing with ema_update

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

        # TODO: nce = update_feature, for loop over feature_update_ratio

        self.rng, self.critic, critic_metrics = update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor_target,
            batch,
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
                batch,
            )
            metrics.update(actor_metrics)

        if self._n_training_steps % self.target_update_freq == 0:
            self.critic_target = ema_update(
                self.critic, self.critic_target, self.cfg.ema
            )
            self.actor_target = ema_update(self.actor, self.actor_target, self.cfg.ema)

        self._n_training_steps += 1
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        assert num_samples == 1, "TD3 only supports num_samples=1"
        self.rng, sample_rng = jax.random.split(self.rng)
        action = jit_sample_action(
            sample_rng,
            self.actor,
            obs,
            deterministic,
            exploration_noise=self.exploration_noise,
        )
        return action, {}
