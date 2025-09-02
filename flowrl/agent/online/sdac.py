from functools import partial
from typing import Tuple

import distrax
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.mujoco.algo.sdac import SDACConfig
from flowrl.flow.ddpm import DDPM, DDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedGaussianActor
from flowrl.module.critic import EnsembleCritic
from flowrl.module.misc import TunableCoefficient
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "num_samples", "solver", "deterministic", "noise_scaler"))
def jit_sample_actions(
    rng: PRNGKey,
    model: DDPM,
    q: Model,
    log_alpha: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    solver: str,
    deterministic: bool,
    noise_scaler: float,
) -> Tuple[PRNGKey, jnp.ndarray]:
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], model.x_dim))
    rng, actions, _ = model.sample(rng, xT, obs_repeat, training, solver)
    if num_samples == 1:
        actions = actions[:, 0]
    else:
        qs = q(obs_repeat, actions)
        qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    if not deterministic:
        rng, stoc_rng = jax.random.split(rng)
        # actions = actions + jax.random.normal(stoc_rng, actions.shape) * jnp.exp(log_alpha()) * noise_scaler
        actions = actions + jax.random.normal(stoc_rng, actions.shape) * 0.1
        actions = jnp.clip(actions, -1.0, 1.0)
    return rng, actions

@partial(jax.jit, static_argnames=("discount", "num_samples", "solver", "noise_scaler", "num_reverse_samples", "target_entropy", "alpha",  "ema"))
def jit_update_sdac(
    rng: PRNGKey,
    actor: DDPM,
    actor_target: DDPM,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    batch: Batch,
    discount: float,
    num_samples: int,
    solver: str,
    noise_scaler: float,
    num_reverse_samples: int,
    target_entropy: float,
    alpha: float,
    q_scale: float,
    ema: float,
) -> Tuple[PRNGKey, DDPM, DDPM, Model, Model, Model, Metric]:

    # update critic
    rng, next_action = jit_sample_actions(
        rng,
        actor,
        critic,
        log_alpha,
        batch.next_obs,
        training=False,
        num_samples=num_samples,
        solver=solver,
        deterministic=False,
        noise_scaler=noise_scaler,
    )
    q_target = critic_target(batch.next_obs, next_action)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target.min(axis=0)

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((q - q_target[jnp.newaxis, :])**2).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

    # update actor
    a0 = next_action
    rng, reverse_rng = jax.random.split(rng)
    rng, at, t, eps = actor.add_noise(rng, a0)
    at = at[jnp.newaxis, ...].repeat(num_reverse_samples, axis=0)
    t = t[jnp.newaxis, ...].repeat(num_reverse_samples, axis=0)
    next_obs_repeat = batch.next_obs[jnp.newaxis, ...].repeat(num_reverse_samples, axis=0)
    # reverse sampling
    eps_reverse = jax.random.normal(reverse_rng, at.shape)
    a0_hat = jnp.sqrt(1 / actor.alpha_hats[t]) * at + jnp.sqrt(1 / actor.alpha_hats[t] - 1) * eps_reverse
    q0 = critic(next_obs_repeat, a0_hat).min(axis=0)
    weights = jax.nn.softmax(q0 / q_scale / alpha, axis=0)
    # Z = jax.nn.logsumexp(q0, axis=0, keepdims=True) - jnp.log(num_reverse_samples) # partition function
    # weights = jnp.exp((q0 - Z) / q_scale / alpha)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": actor_params},
            at,
            t,
            condition=next_obs_repeat,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = (weights * ((eps_pred - eps_reverse) ** 2)).mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/weights": weights.mean(),
            "misc/weight_std": weights.std(0).mean(),
            "misc/weights_max": weights.max(),
            "misc/weights_min": weights.min(),
            "misc/q_scale": q_scale,
        }

    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

    # update log_alpha
    approx_entropy = 0.5 * next_action.shape[-1] * jnp.log(2*jnp.pi*jnp.e*jnp.exp(log_alpha()))
    def alpha_loss_fn(log_alpha_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        log_alpha_value = log_alpha.apply(
            {"params": log_alpha_params},
        )
        loss = log_alpha_value * jax.lax.stop_gradient(approx_entropy - target_entropy)
        return loss, {
            "loss/alpha_loss": loss,
            "misc/approx_entropy": approx_entropy,
            "misc/alpha": jnp.exp(log_alpha_value).mean(),
        }
    new_log_alpha, alpha_metrics = log_alpha.apply_gradient(alpha_loss_fn)

    new_actor_target = ema_update(new_actor, actor_target, ema)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    new_q_scale = q_scale + ema * (q0.std(axis=0).mean() - q_scale)
    return rng, new_actor, new_actor_target, new_critic, new_critic_target, new_log_alpha, new_q_scale, {
        **critic_metrics,
        **actor_metrics,
        **alpha_metrics,
    }


class SDACAgent(BaseAgent):
    """
    Soft Diffusion Actor-Critic (SDAC) agent.
    """
    name = "SDACAgent"
    model_names = [] # TODO

    def __init__(self, obs_dim: int, act_dim: int, cfg: SDACConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng, alpha_rng = jax.random.split(self.rng, 4)

        # define the actor
        time_embedding = partial(PositionalEmbedding, output_dim=cfg.diffusion.time_dim)
        noise_predictor = partial(
            MLP,
            hidden_dims=cfg.diffusion.mlp_hidden_dims,
            output_dim=act_dim,
            activation=mish,
            layer_norm=False,
            dropout=None,
        )
        backbone_def = DDPMBackbone(
            noise_predictor=noise_predictor,
            time_embedding=time_embedding,
        )

        if cfg.diffusion.lr_decay_steps is not None:
            actor_lr = optax.linear_schedule(
                init_value=cfg.diffusion.lr,
                end_value=cfg.diffusion.end_lr,
                transition_steps=cfg.diffusion.lr_decay_steps,
                transition_begin=cfg.diffusion.lr_decay_begin,
            )
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
            approx_postvar=False,
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            optimizer=optax.adam(learning_rate=actor_lr),
        )
        # CHECK: is this really necessary, since we are not using the target actor for policy evaluation?
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

        # define the critic
        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic_hidden_dims,
            activation=mish,
            layer_norm=False,
            dropout=None,
            ensemble_size=2,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )
        # define entropy related terms
        self.alpha = cfg.alpha
        self.q_scale = jnp.array(1.0)
        self.log_alpha = Model.create(
            TunableCoefficient(init_value=0.2),
            alpha_rng,
            inputs=(),
            optimizer=optax.adam(learning_rate=cfg.alpha_lr),
        )
        self.target_entropy = - act_dim * self.cfg.target_entropy_scale

        # define tracking variables
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.actor_target, self.critic, self.critic_target, self.log_alpha, self.q_scale, metrics = jit_update_sdac(
            self.rng,
            self.actor,
            self.actor_target,
            self.critic,
            self.critic_target,
            self.log_alpha,
            batch,
            discount=self.cfg.discount,
            num_samples=self.cfg.num_samples,
            solver=self.cfg.diffusion.solver,
            noise_scaler=self.cfg.diffusion.noise_scaler,
            num_reverse_samples=self.cfg.num_reverse_samples,
            target_entropy=self.target_entropy,
            alpha=self.alpha,
            q_scale=self.q_scale,
            ema=self.cfg.ema,
        )
        self._n_training_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        self.rng, action = jit_sample_actions(
            self.rng,
            self.actor_target,
            self.critic,
            self.log_alpha,
            obs,
            training=False,
            num_samples=self.cfg.num_samples, # NOTE: we sample multiple actions and get the best one
            solver=self.cfg.diffusion.solver,
            deterministic=deterministic,
            noise_scaler=self.cfg.diffusion.noise_scaler,
        )
        return action, {}
