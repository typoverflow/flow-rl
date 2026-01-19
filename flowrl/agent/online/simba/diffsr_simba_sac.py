from functools import partial

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.simba.diffsr_network import (
    FactorizedDDPM,
    update_factorized_ddpm,
)
from flowrl.agent.online.simba.network import EnsembleSimbaCritic, SimbaNet
from flowrl.agent.online.simba.simba_sac import SimbaSACAgent
from flowrl.config.online.algo.simba.diffsr_simba_sac import DiffSRSimbaSACConfig
from flowrl.functional.ema import ema_update
from flowrl.module.model import Model
from flowrl.types import *
from flowrl.types import Batch, Metric, Param, PRNGKey


def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor: Model,
    alpha: Model,
    ddpm_target: Model,
    batch: Batch,
    discount: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    dist = actor(batch.next_obs)
    next_action, next_logprob = dist.sample_and_log_prob(seed=sample_rng)
    next_feature = ddpm_target(batch.next_obs, next_action, method="forward_phi")
    next_q = critic_target(next_feature)
    target_q = batch.reward + discount * (1-batch.terminal) * (
        next_q.min(axis=0) \
        - alpha() * next_logprob
    )

    feature = ddpm_target(batch.obs, batch.action, method="forward_phi")
    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            feature,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((q - target_q[jnp.newaxis, :])**2).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q.mean(),
            "misc/reward": batch.reward.mean(),
        }
    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics

def update_actor(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    alpha: Model,
    ddpm_target: Model,
    batch: Batch,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        dist = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        new_action, new_logprob = dist.sample_and_log_prob(seed=sample_rng)
        new_feature = ddpm_target(batch.obs, new_action, method="forward_phi")
        q = critic(new_feature)
        actor_loss = (alpha() * new_logprob - q.mean(axis=0)).mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss,
        }
    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, metrics

def update_alpha(
    rng: PRNGKey,
    alpha: Model,
    actor: Model,
    batch: Batch,
    target_entropy: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    dist = actor(batch.obs)
    action, logprob = dist.sample_and_log_prob(seed=sample_rng)

    def alpha_loss_fn(alpha_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        alpha_value = alpha.apply(
            {"params": alpha_params},
        )
        loss = - alpha_value * (logprob + target_entropy).mean()
        return loss, {
            "loss/alpha_loss": loss,
            "misc/alpha": alpha_value,
            "misc/entropy": -logprob.mean(),
        }
    new_alpha, metrics = alpha.apply_gradient(alpha_loss_fn)
    return rng, new_alpha, metrics

def update_once(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    ddpm: Model,
    ddpm_target: Model,
    alpha: Model,
    batch: Batch,
    discount: float,
    ema: float,
    target_entropy: float,
    reward_coef: float,
    terminal_coef: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Metric]:
    rng, new_ddpm, ddpm_metrics = update_factorized_ddpm(rng, ddpm, batch, reward_coef, terminal_coef)
    rng, new_critic, critic_metrics = update_critic(rng, critic, critic_target, actor, alpha, ddpm_target, batch, discount)
    rng, new_actor, actor_metrics = update_actor(rng, actor, new_critic, alpha, ddpm_target, batch)
    rng, new_alpha, alpha_metrics = update_alpha(rng, alpha, new_actor, batch, target_entropy)
    new_ddpm_target = ema_update(new_ddpm, ddpm_target, ema)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    return rng, new_actor, new_critic, new_critic_target, new_alpha, new_ddpm, new_ddpm_target, {
        **critic_metrics,
        **actor_metrics,
        **alpha_metrics,
        **ddpm_metrics,
    }

@partial(jax.jit, static_argnames=("num_updates", "discount", "ema", "target_entropy", "reward_coef", "terminal_coef"))
def update_multiple(
    num_updates: int,
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    alpha: Model,
    ddpm: Model,
    ddpm_target: Model,
    batch: Batch,
    discount: float,
    ema: float,
    target_entropy: float,
    reward_coef: float,
    terminal_coef: float,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, Metric]:
    mini_batch_size = batch.obs.shape[0] // num_updates
    batch = jax.tree.map(lambda x: x.reshape((num_updates, mini_batch_size, -1)) if x is not None else None, batch)
    def one_update(i, state):
        rng, actor, critic, critic_target, alpha, ddpm, ddpm_target, metrics = state
        new_rng, new_actor, new_critic, new_critic_target, new_alpha, new_ddpm, new_ddpm_target, new_metrics = update_once(
            rng, actor, critic, critic_target, ddpm, ddpm_target, alpha, jax.tree.map(lambda x: jnp.take(x, i, axis=0) if x is not None else None, batch), discount, ema, target_entropy, reward_coef, terminal_coef)
        return new_rng, new_actor, new_critic, new_critic_target, new_alpha, new_ddpm, new_ddpm_target, new_metrics
    rng, actor, critic, critic_target, alpha, ddpm, ddpm_target, metrics = one_update(0, (rng, actor, critic, critic_target, alpha, ddpm, ddpm_target, {}))
    return jax.lax.fori_loop(1, num_updates, one_update, (rng, actor, critic, critic_target, alpha, ddpm, ddpm_target, metrics))


class DiffSRSimbaSACAgent(SimbaSACAgent):
    """
    DiffSR Simba Soft Actor-Critic (SAC) agent.
    """
    name = "DiffSRSimbaSACAgent"
    model_names = ["ddpm", "ddpm_target", "actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DiffSRSimbaSACConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, ddpm_rng, ddpm_init_rng, actor_rng, critic_rng = jax.random.split(self.rng, 5)

        self.num_updates = cfg.num_updates

        ddpm_def = FactorizedDDPM(
            self.obs_dim,
            self.act_dim,
            cfg.feature_dim,
            cfg.embed_dim,
            cfg.phi_hidden_dims,
            cfg.mu_hidden_dims,
            cfg.reward_hidden_dims,
            cfg.rff_dim,
            cfg.num_noises,
        )
        self.ddpm = Model.create(
            ddpm_def,
            ddpm_rng,
            inputs=(
                ddpm_init_rng,
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
            optimizer=optax.adam(learning_rate=cfg.feature_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.ddpm_target = Model.create(
            ddpm_def,
            ddpm_rng,
            inputs=(
                ddpm_init_rng,
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
        )
        critic_def = EnsembleSimbaCritic(
            num_blocks=cfg.critic_num_blocks,
            hidden_dim=cfg.critic_hidden_dim,
            ensemble_size=cfg.critic_ensemble_size,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.cfg.feature_dim)), ),
            optimizer=optax.adamw(learning_rate=cfg.critic_lr, weight_decay=cfg.critic_wd),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.cfg.feature_dim)), ),
        )

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.critic, self.critic_target, self.alpha, self.ddpm, self.ddpm_target, metrics = update_multiple(
            self.num_updates,
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            self.alpha,
            self.ddpm,
            self.ddpm_target,
            batch=batch,
            discount=self.cfg.discount,
            ema=self.cfg.ema,
            target_entropy=self.target_entropy,
            reward_coef=self.cfg.reward_coef,
            terminal_coef=self.cfg.terminal_coef,
        )
        return metrics
