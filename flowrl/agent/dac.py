from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.d4rl.algo.dac import DACConfig
from flowrl.flow.ddpm import DDPM, DDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import EnsembleCritic
from flowrl.module.mlp import MLP, ResidualMLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import *

EPS = 1e-6

@partial(jax.jit, static_argnames=("maxQ", "q_target", "rho"))
def get_target(rng: PRNGKey, vs: jnp.ndarray, maxQ: bool, q_target: str, rho: float) -> Tuple[PRNGKey, jnp.ndarray]:
    # vs is of shape (E, B, N, 1)
    if maxQ:
        vs = vs.max(axis=-2)
    else:
        vs = vs.mean(axis=-2)
    if q_target == "min":
        vs = vs.min(axis=0)
    elif q_target == "convex":
        vs = rho * vs.min(axis=0) + (1-rho) * vs.max(axis=0)
    elif q_target == "lcb":
        vs = vs.mean(axis=0) - rho * vs.std(axis=0)
    elif q_target == "rand_convex":
        rng, alpha_key = jax.random.split(rng)
        alphas = jax.random.uniform(alpha_key, vs.shape)
        alphas /= (alphas.sum(axis=0, keepdims=True) + EPS)
        vs = (vs * alphas).sum(axis=0)
    else:
        raise NotImplementedError(f"Unrecognized Q-target type: {q_target}. ")
    return rng, vs

@partial(jax.jit, static_argnames=("discount", "rho", "num_samples", "q_target", "maxQ", "solver", "ema", "do_ema_update"))
def jit_update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_target: DDPM,
    batch: Batch,
    discount: float,
    rho: float,
    num_samples: int,
    q_target: str,
    maxQ: bool,
    solver: str,
    ema: float,
    do_ema_update: bool
) -> Tuple[PRNGKey, Model, Model, Metric]:
    B = batch.obs.shape[0]
    A = batch.action.shape[-1]

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
    rng, next_q = get_target(rng, next_q, maxQ, q_target, rho)
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
    if do_ema_update:
        new_critic_target = ema_update(new_critic, critic_target, ema)
    else:
        new_critic_target = critic_target
    return rng, new_critic, new_critic_target, critic_metrics

@partial(jax.jit, static_argnames=("eta_min", "eta_max", "eta_lr", "eta_threshold", "ema", "do_ema_update"))
def jit_update_actor(
    rng: PRNGKey,
    actor: DDPM,
    actor_target: DDPM,
    critic_target: Model,
    eta: jnp.ndarray,
    batch: Batch,
    eta_min: float,
    eta_max: float,
    eta_lr: float,
    eta_threshold: float,
    ema: float,
    do_ema_update: bool
) -> Tuple[PRNGKey, Model, Model, jnp.ndarray, Metric]:
    B = batch.obs.shape[0]

    rng, xt, t, eps = actor.add_noise(rng, batch.action)
    q = critic_target(batch.obs, batch.action)
    q_norm = jnp.abs(q).mean()
    q_grad_fn = jax.vmap(jax.grad(lambda a, s: critic_target(s, a).mean()))
    q_grad = q_grad_fn(xt, batch.obs) / (q_norm + EPS)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": actor_params},
            xt,
            t,
            condition=batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        bc_loss = ((eps_pred - eps) ** 2).mean()
        guidance_loss = (jnp.sqrt(1-actor.alpha_hats[t]) * q_grad * eps_pred).mean()
        actor_loss = eta * bc_loss + guidance_loss
        return actor_loss, {
            "loss/actor_loss": actor_loss,
            "loss/bc_loss": bc_loss,
            "loss/guidance_loss": guidance_loss,
            "misc/eta": eta,
            "misc/q_grad_abs": jnp.abs(q_grad).mean(),
        }
    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)
    if do_ema_update:
        new_actor_target = ema_update(new_actor, actor_target, ema)
    else:
        new_actor_target = actor_target

    if eta_lr > 0:
        bc_loss = actor_metrics["loss/bc_loss"]
        eta += eta_lr * (bc_loss - eta_threshold).clip(-1, 1)
        eta = jnp.clip(eta, eta_min, eta_max)

    return rng, new_actor, new_actor_target, eta, actor_metrics

@partial(jax.jit, static_argnames=("training", "num_samples", "solver", "temperature"))
def jit_sample_and_select(
    rng: PRNGKey,
    model: DDPM,
    q0: Model,
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
        qs = q0(
            obs_repeat,
            actions,
        )
        qs = qs.mean(axis=0).reshape(B, num_samples)
        if temperature <= 0.0:
            idx = qs.argmax(axis=-1)
        else:
            rng, select_rng = jax.random.split(rng)
            idx = jax.random.categorical(select_rng, logits=qs/(1e-6+temperature), axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), idx]
        return rng, actions


class DACAgent(BaseAgent):
    """
    Diffusion Actor Critic
    """
    name = "DACAgent"
    model_names = ["actor", "actor_target", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DACConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        # define the actor
        time_embedding = partial(PositionalEmbedding, output_dim=cfg.diffusion.time_dim)
        if cfg.diffusion.resnet:
            noise_predictor = partial(
                ResidualMLP,
                hidden_dims=cfg.diffusion.resnet_hidden_dims,
                output_dim=act_dim,
                activation=mish,
                layer_norm=True,
                dropout=cfg.diffusion.dropout
            )
        else:
            noise_predictor = partial(
                MLP,
                hidden_dims=cfg.diffusion.mlp_hidden_dims,
                output_dim=act_dim,
                activation=mish,
                layer_norm=cfg.diffusion.layer_norm,
                dropout=cfg.diffusion.dropout
            )
        backbone_def = DDPMBackbone(
            noise_predictor=noise_predictor,
            time_embedding=time_embedding
        )
        if cfg.diffusion.lr_decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(cfg.diffusion.lr, cfg.diffusion.lr_decay_steps)
        else:
            actor_lr = cfg.diffusion.lr
        self.actor = DDPM.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule=cfg.diffusion.noise_schedule,
            noise_schedule_params=None,
            approx_postvar=True,
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            optimizer=optax.adam(learning_rate=actor_lr),
            clip_grad_norm=cfg.diffusion.clip_grad_norm,
        )
        self.actor_target = DDPM.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule=cfg.diffusion.noise_schedule,
            noise_schedule_params=None,
            approx_postvar=True,
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
        )

        # define the critics
        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic.hidden_dims,
            activation=mish,
            ensemble_size=cfg.critic.ensemble_size,
            layer_norm=cfg.critic.layer_norm,
        )
        if cfg.critic.lr_decay_steps is not None:
            critic_lr = optax.cosine_decay_schedule(cfg.critic.lr, cfg.critic.lr_decay_steps)
        else:
            critic_lr = cfg.critic.lr
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=critic_lr),
            clip_grad_norm=cfg.critic.clip_grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        self.eta = jnp.array(cfg.eta)

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
            self.cfg.critic.rho,
            self.cfg.critic.num_samples,
            self.cfg.critic.q_target,
            self.cfg.critic.maxQ,
            self.cfg.diffusion.solver,
            self.cfg.critic.ema,
            self._n_training_steps % self.cfg.critic.ema_every == 0,
        )
        self.rng, self.actor, self.actor_target, self.eta, actor_metrics = jit_update_actor(
            self.rng,
            self.actor,
            self.actor_target,
            self.critic_target,
            self.eta,
            batch,
            self.cfg.eta_min,
            self.cfg.eta_max,
            self.cfg.eta_lr,
            self.cfg.eta_threshold,
            self.cfg.diffusion.ema,
            self._n_training_steps >= self.cfg.start_actor_ema and \
            self._n_training_steps % self.cfg.diffusion.ema_every == 0,
        )
        metrics.update(critic_metrics)
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
            temperature=self.cfg.temperature,
        )
        return action, {}
