from functools import partial
from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.d4rl.algo.dtql import DTQLConfig
from flowrl.flow.edm import EDM, EDMBackbone, compute_edm_loss
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.functional.loss import expectile_regression
from flowrl.module.actor import SquashedGaussianActor
from flowrl.module.critic import Critic, EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey
from flowrl.utils.distribution import TanhMultivariateNormalDiag


@partial(jax.jit, static_argnames=("num_samples"))
def jit_sample_action(
    rng: PRNGKey,
    actor: Model,
    q: Model,
    obs: jnp.ndarray,
    num_samples: int,
) -> Tuple[PRNGKey, jnp.ndarray]:
    """
    Sample action from the actor network.
    """
    B = obs.shape[0]
    rng, sample_key = jax.random.split(rng)
    dist: TanhMultivariateNormalDiag = actor(obs)
    action = dist.sample(seed=jax.random.PRNGKey(0), sample_shape=(num_samples,)) # [num_samples, B, act_dim]
    q_value_ensembled = q(obs[jnp.newaxis, ...].repeat(num_samples, axis=0), action) # [2, num_samples, B, 1]
    q_value = jnp.min(q_value_ensembled, axis=0).squeeze(-1) # [num_samples, B]
    idx = jax.random.categorical(sample_key, nn.softmax(q_value, axis=0), axis=0) # [B]
    action = action[idx, jnp.arange(B)] # [B, act_dim]
    return rng, action

@jax.jit
def jit_update_bc(
    rng: PRNGKey,
    bc_actor: EDM,
    batch: Batch,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, loss_key = jax.random.split(rng)
    def bc_loss_fn(params: Param, dropout_rng: PRNGKey):
        loss = compute_edm_loss(
            loss_key,
            bc_actor,
            batch.action,
            batch.obs,
            training=True,
            params=params,
            dropout_rng=dropout_rng,
        )[1]
        return loss, {
            "bc_loss": loss,
        }
    new_bc_actor, metrics = bc_actor.apply_gradient(bc_loss_fn)
    return rng, new_bc_actor, metrics

@partial(jax.jit, static_argnames=("discount", "expectile", "metric_prefix", "ema", "do_ema_update"))
def jit_update_critic(
    q_net: Model,
    q_net_target: Model,
    value: Model,
    value_target: Model,
    batch: Batch,
    discount: float,
    expectile: float,
    metric_prefix: str,
    ema: float,
    do_ema_update: bool,
) -> Tuple[Model, Model, Model, Model, Metric]:
    metric_prefix = f"{metric_prefix}_" if metric_prefix else ""
    ensemble_q = q_net(batch.obs, batch.action) # [2, B, 1]
    q = jnp.min(ensemble_q, axis=0) # [B, 1]
    def value_loss_fn(params: Param, dropout_rng: PRNGKey):
        pred = value.apply({"params": params}, batch.obs, training=True, rngs={"dropout": dropout_rng}) # [B, 1]
        value_loss = expectile_regression(pred, q, expectile=expectile).mean()
        return value_loss, {
            f"{metric_prefix}loss/value_loss": value_loss,
            f"misc/{metric_prefix}value_mean": pred.mean(),
        }
    new_value, value_metrics = value.apply_gradient(value_loss_fn)

    next_v = value(batch.next_obs) # [B, 1], DTQL's official implementation didn't use target value network here
    target_q = batch.reward + discount * (1 - batch.terminal) * next_v # [B, 1]
    def q_loss_fn(params: Param, dropout_rng: PRNGKey):
        pred = q_net.apply({"params": params}, batch.obs, batch.action, training=True, rngs={"dropout": dropout_rng}) # [2, B, 1]
        q_loss = ((pred - target_q[jnp.newaxis, :])**2).sum(axis=0).mean()
        return q_loss, {
            f"{metric_prefix}loss/q_loss": q_loss,
            f"misc/{metric_prefix}q_mean": pred.mean(),
        }
    new_q_net, q_metrics = q_net.apply_gradient(q_loss_fn)

    if do_ema_update:
        new_q_net_target = ema_update(new_q_net, q_net_target, ema)
        new_value_target = ema_update(new_value, value_target, ema)
    else:
        new_q_net_target = q_net_target
        new_value_target = value_target
    return new_q_net, new_q_net_target, new_value, new_value_target, {
        **value_metrics,
        **q_metrics,
        f"misc/{metric_prefix}reward": batch.reward.mean(),
    }

@partial(jax.jit, static_argnames=("gamma", "alpha"))
def jit_update_distill_actor(
    rng: PRNGKey,
    distill_actor: Model,
    bc_actor: EDM,
    q_net: Model,
    batch: Batch,
    gamma: float,
    alpha: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, loss_key = jax.random.split(rng)
    def loss_fn(params: Param, dropout_rng: PRNGKey):
        dist: TanhMultivariateNormalDiag = distill_actor.apply({"params": params}, batch.obs, training=True, rngs={"dropout": dropout_rng})
        new_action, log_prob = dist.sample_and_log_prob(seed=jax.random.PRNGKey(0))
        distill_loss = compute_edm_loss(
            loss_key,
            bc_actor,
            new_action,
            batch.obs,
        )[1]
        gamma_loss = -log_prob.mean()
        q_loss = -q_net(batch.obs, new_action).min(axis=0).mean()
        loss = alpha * distill_loss + gamma * gamma_loss + q_loss
        return loss, {
            "loss/distill_distill_loss": distill_loss,
            "loss/distill_gamma_loss": gamma_loss,
            "loss/distill_q_loss": q_loss,
            "loss/distill_total_loss": loss,
            "misc/log_prob": -log_prob.mean(),
        }
    new_distill_actor, metrics = distill_actor.apply_gradient(loss_fn)
    return rng, new_distill_actor, metrics

CRITIC_LR = 3e-4 # DTQL hard codes the critic lr

class DTQLAgent(BaseAgent):
    """
    Diffusion Trusted Q-Learning (DTQL, https://arxiv.org/abs/2405.19690) agent.
    """
    name = "DTQLAgent"
    model_names = ["q", "q_target", "value", "value_target", "bc_actor", "distill_actor"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DTQLConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, bc_key, actor_key, critic_key, value_key = jax.random.split(self.rng, 5)

        q_def = EnsembleCritic(
            ensemble_size=2,
            hidden_dims=[256,256,256], # DTQL hard codes the critic hidden dims
            activation=mish,
            layer_norm=True,
        )
        self.q = Model.create(
            q_def,
            rng=critic_key,
            inputs=(jnp.ones((1, obs_dim)), jnp.ones((1, act_dim))),
            optimizer=optax.adam(CRITIC_LR),
        )
        self.q_target = Model.create(
            q_def,
            rng=critic_key,
            inputs=(jnp.ones((1, obs_dim)), jnp.ones((1, act_dim))),
        )

        v_def = Critic(
            hidden_dims=[256,256,256], # DTQL hard codes the critic hidden dims
            activation=mish,
        )
        self.value = Model.create(
            v_def,
            rng=value_key,
            inputs=(jnp.ones((1, obs_dim)),),
            optimizer=optax.adam(CRITIC_LR),
        )
        self.value_target = Model.create(
            v_def,
            rng=value_key,
            inputs=(jnp.ones((1, obs_dim)),),
        )

        # DTQL's official implementation creates targets for bc_actor and distill_actor but NEVER USES THEM. So we don't create them here.
        self.bc_actor = EDM.create(
            EDMBackbone(
                noise_predictor=partial(
                    MLP,
                    hidden_dims=[256, 256, 256, 256], # DTQL hard codes the backbone hidden dims
                    output_dim=act_dim,
                    activation=mish,
                ),
                time_embedding=partial(PositionalEmbedding, output_dim=16), # DTQL hard codes the time embedding output dim
            ),
            rng=bc_key,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim))),
            x_dim=self.act_dim,
            steps=0, # we don't need to sample from BC actor
            sigma_sample_density_type="loglogistic",
            sigma_data=cfg.sigma_data,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            clip_sampler=True,
            x_min=self.cfg.min_action,
            x_max=self.cfg.max_action,
            optimizer=optax.adam(cfg.lr),
        )
        self.distill_actor = Model.create(
            SquashedGaussianActor(
                backbone=MLP(hidden_dims=[256, 256, 256]), # DTQL hard codes the actor hidden dims
                obs_dim=obs_dim,
                action_dim=act_dim,
                conditional_logstd=True,
                logstd_min=-5,
                logstd_max=2,
            ),
            rng=actor_key,
            inputs=(jnp.ones((1, obs_dim)),),
            optimizer=optax.adam(cfg.lr),
        )

    def prepare_training(self):
        # DTQL's official implementation don't use lr schedule during pretraining,
        # so we reset optimizers after pretraining and before training
        if self.cfg.lr_decay:
            self.q = self.q.reset_optim(optax.adam(optax.cosine_decay_schedule(CRITIC_LR, self.cfg.lr_decay_steps)))
            self.value = self.value.reset_optim(optax.adam(optax.cosine_decay_schedule(CRITIC_LR, self.cfg.lr_decay_steps)))
            self.bc_actor = self.bc_actor.reset_optim(optax.adam(optax.cosine_decay_schedule(self.cfg.lr, self.cfg.lr_decay_steps)))
            self.distill_actor = self.distill_actor.reset_optim(optax.adam(optax.cosine_decay_schedule(self.cfg.lr, self.cfg.lr_decay_steps)))

    def pretrain_step(self, batch, step):
        self.rng, self.bc_actor, bc_metrics = jit_update_bc(
            self.rng,
            self.bc_actor,
            batch,
        )
        bc_metrics = {f"pretrain_loss/{k}": v for k, v in bc_metrics.items()}
        self.q, self.q_target, self.value, self.value_target, critic_metrics = jit_update_critic(
            self.q,
            self.q_target,
            self.value,
            self.value_target,
            batch,
            discount=self.cfg.discount,
            expectile=self.cfg.expectile,
            metric_prefix="pretrain",
            ema=self.cfg.ema,
            do_ema_update=False, # no ema update during pretraining
        )
        return {
            **bc_metrics,
            **critic_metrics,
        }

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.bc_actor, bc_metrics = jit_update_bc(
            self.rng,
            self.bc_actor,
            batch,
        )
        bc_metrics = {f"loss/{k}": v for k, v in bc_metrics.items()}
        self.q, self.q_target, self.value, self.value_target, critic_metrics = jit_update_critic(
            self.q,
            self.q_target,
            self.value,
            self.value_target,
            batch,
            discount=self.cfg.discount,
            expectile=self.cfg.expectile,
            metric_prefix="",
            ema=self.cfg.ema,
            do_ema_update=step % 5 == 0, # DTQL's official implementation hard codes the ema update to be every 5 steps
        )
        self.rng, self.distill_actor, distill_metrics = jit_update_distill_actor(
            self.rng,
            self.distill_actor,
            self.bc_actor,
            self.q,
            batch,
            gamma=self.cfg.gamma,
            alpha=self.cfg.alpha,
        )
        return {
            **bc_metrics,
            **critic_metrics,
            **distill_metrics,
        }

    def sample_actions(self, obs, deterministic=True, num_samples = 1):
        # NOTE: DTQL's official implementation always use deterministic=False
        self.rng, action = jit_sample_action(
            self.rng,
            self.distill_actor,
            self.q_target,
            obs,
            num_samples=num_samples,
        )
        return action, {}
