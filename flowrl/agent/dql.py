from functools import partial

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.d4rl.algo.dql import DQLConfig
from flowrl.flow.ddpm import DDPM, DDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import *


@partial(jax.jit, static_argnames=("discount", "maxQ", "solver", "ema"))
def jit_update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_target: DDPM,
    batch: Batch,
    discount: float,
    maxQ: bool,
    solver: str,
    ema: float,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    B = batch.obs.shape[0]
    A = batch.action.shape[-1]

    num_samples = 10 if maxQ else 1
    rng, xT_rng = jax.random.split(rng)
    next_obs_repeat = batch.next_obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*next_obs_repeat.shape[:-1], A))
    rng, next_action, history = actor_target.sample(
        rng,
        xT,
        next_obs_repeat,
        training=False,
        solver=solver
    )
    next_q = critic_target(next_obs_repeat, next_action)
    if maxQ:
        next_q = next_q.max(axis=-2)
    else:
        next_q = next_q.mean(axis=-2)

    next_q = next_q.min(axis=0)
    next_q = batch.reward + discount * (1-batch.terminal) * next_q

    def critic_loss_fn(critic_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        pred = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action
        )
        critic_loss = ((pred - next_q[jnp.newaxis, :])**2).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": pred.mean(),
            "misc/reward": batch.reward.mean()
        }
    new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)
    new_target_critic = ema_update(new_critic, critic_target, ema)

    return rng, new_critic, new_target_critic, critic_metrics

@partial(jax.jit, static_argnames=("eta", "solver", "ema", "do_ema_update"))
def jit_update_actor(
    rng: PRNGKey,
    actor: DDPM,
    actor_target: DDPM,
    critic: Model,
    batch: Batch,
    eta: float,
    solver: str,
    ema: float,
    do_ema_update: bool,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    B = batch.obs.shape[0]
    A = batch.action.shape[-1]
    rng, xT_rng, choice_rng = jax.random.split(rng, 3)

    # for bc loss
    rng, xt, t, eps = actor.add_noise(rng, batch.action)

    # for q loss
    xT = jax.random.normal(xT_rng, (*batch.obs.shape[:-1], A))

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        rng1, rng2 = jax.random.split(dropout_rng)
        eps_pred = actor.apply(
            {"params": actor_params},
            xt,
            t,
            condition=batch.obs,
            training=True,
            rngs={"dropout": rng1},
        )
        bc_loss = ((eps_pred - eps) ** 2).mean()

        _, new_action, _ = actor.sample(
            rng2,
            xT,
            batch.obs,
            training=True,
            solver=solver,
            params=actor_params,
        )
        new_q = critic(batch.obs, new_action)
        choice = jax.random.uniform(choice_rng)
        q_loss1 = - new_q[0].mean() / jax.lax.stop_gradient(jnp.abs(new_q[1]).mean() + 1e-6)
        q_loss2 = - new_q[1].mean() / jax.lax.stop_gradient(jnp.abs(new_q[0]).mean() + 1e-6)
        q_loss = (choice > 0.5) * q_loss1 + (choice <= 0.5) * q_loss2
        actor_loss = bc_loss + q_loss * eta
        return actor_loss, {
            "loss/bc_loss": bc_loss,
            "misc/eta": eta,
        }
    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)
    if do_ema_update:
        new_actor_target = ema_update(new_actor, actor_target, ema)
    else:
        new_actor_target = actor_target
    return rng, new_actor, new_actor_target, actor_metrics

@partial(jax.jit, static_argnames=("training", "num_samples", "solver", "temperature"))
def jit_sample_and_select(
    rng: PRNGKey,
    model: DDPM,
    critic: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    solver: str,
    temperature: float,
) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], model.x_dim))
    rng, actions, _ = model.sample(rng, xT, obs_repeat, training, solver)
    if temperature is None:
        return rng, actions[:, 0]
    else:
        qs = critic(
            obs_repeat,
            actions,
        )
        qs = qs.min(axis=0).reshape(B, num_samples)
        if temperature <= 0.0:
            idx = qs.argmax(axis=-1)
        else:
            rng, select_rng = jax.random.split(rng)
            idx = jax.random.categorical(select_rng, logits=qs/(1e-6+temperature), axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), idx]
        return rng, actions


class DQLAgent(BaseAgent):
    """
    Diffusion Q-Learning
    """
    name = "DQLAgent"
    model_names = ["actor", "actor_target", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DQLConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        if cfg.lr_decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(cfg.lr, cfg.lr_decay_steps)
            critic_lr = optax.cosine_decay_schedule(cfg.lr, cfg.lr_decay_steps)
        else:
            actor_lr = cfg.lr
            critic_lr = cfg.lr
        # define the actor
        time_embedding = partial(PositionalEmbedding, output_dim=cfg.diffusion.time_dim)
        noise_predictor = partial(
            MLP,
            hidden_dims=cfg.diffusion.hidden_dims,
            output_dim=act_dim,
            activation=mish,
        )
        backbone_def = DDPMBackbone(
            noise_predictor=noise_predictor,
            time_embedding=time_embedding,
        )
        self.actor = DDPM.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule=cfg.diffusion.noise_schedule,
            noise_schedule_params=None,
            approx_postvar=False,
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            optimizer=optax.adam(learning_rate=actor_lr),
            clip_grad_norm=cfg.grad_norm,
        )
        self.actor_target = DDPM.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule=cfg.diffusion.noise_schedule,
            noise_schedule_params=None,
            approx_postvar=False,
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
        )

        # define the critics
        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic.hidden_dims,
            activation=mish,
            ensemble_size=cfg.critic.ensemble_size,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=critic_lr),
            clip_grad_norm=cfg.grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        metrics = {}
        self.rng, self.critic, self.critic_target, critic_metrics = jit_update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor_target,
            batch,
            self.cfg.critic.discount,
            self.cfg.critic.maxQ,
            self.cfg.diffusion.solver,
            self.cfg.critic.ema,
        )
        metrics.update(critic_metrics)

        self.rng, self.actor, self.actor_target, actor_metrics = jit_update_actor(
            self.rng,
            self.actor,
            self.actor_target,
            self.critic,
            batch,
            self.cfg.eta,
            self.cfg.diffusion.solver,
            self.cfg.diffusion.ema,
            self._n_training_steps >= self.cfg.start_actor_ema and \
            self._n_training_steps % self.cfg.diffusion.ema_every == 0,
        )
        metrics.update(actor_metrics)

        self._n_training_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        self.rng, action = jit_sample_and_select(
            self.rng,
            self.actor,
            self.critic_target,
            obs,
            training=False,
            num_samples=num_samples,
            solver=self.cfg.diffusion.solver,
            temperature=self.cfg.temperature
        )
        return action, {}
