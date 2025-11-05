from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.unirep.network import FactorizedNCE, update_factorized_nce
from flowrl.config.online.mujoco.algo.unirep.aca import ACAConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "num_samples", "solver"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic: Model,
    nce_target: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    solver: str,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], actor.x_dim))
    rng, actions, _ = actor.sample(rng, xT, obs_repeat, training, solver)
    if num_samples == 1:
        actions = actions[:, 0]
    else:
        t0 = jnp.ones((obs_repeat.shape[0], num_samples, 1))
        f0 = nce_target(obs_repeat, actions, t0, method="forward_phi")
        qs = critic(f0)
        qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions

@partial(jax.jit, static_argnames=("discount", "solver", "critic_coef"))
def jit_update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor: ContinuousDDPM,
    nce_target: Model,
    batch: Batch,
    discount: float,
    solver: str,
    critic_coef: float,
) -> Tuple[PRNGKey, Model, Metric]:
    # q0 target
    t0 = jnp.ones((batch.obs.shape[0], 1))
    rng, next_aT_rng = jax.random.split(rng)
    next_aT = jax.random.normal(next_aT_rng, (*batch.next_obs.shape[:-1], actor.x_dim))
    rng, next_a0, _ = actor.sample(rng, next_aT, batch.next_obs, training=False, solver=solver)
    next_f0 = nce_target(batch.next_obs, next_a0, t0, method="forward_phi")
    q0_target = critic_target(next_f0)
    q0_target = batch.reward + discount * (1 - batch.terminal) * q0_target.min(axis=0)

    # qt target
    a0 = batch.action
    f0 = nce_target(batch.obs, a0, t0, method="forward_phi")
    qt_target = critic_target(f0)

    # features
    rng, at, t, eps = actor.add_noise(rng, a0)
    ft = nce_target(batch.obs, at, t, method="forward_phi")

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q0_pred = critic.apply(
            {"params": critic_params},
            f0,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        qt_pred = critic.apply(
            {"params": critic_params},
            ft,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = (
            ((q0_pred - q0_target[jnp.newaxis, :])**2).mean() +
            ((qt_pred - qt_target[jnp.newaxis, :])**2).mean()
        )
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q0_mean": q0_pred.mean(),
            "misc/qt_mean": qt_pred.mean(),
            "misc/reward": batch.reward.mean(),
            "misc/next_action_l1": jnp.abs(next_a0).mean(),
        }

    new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, critic_metrics

@partial(jax.jit, static_argnames=("temp",))
def jit_update_actor(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    nce_target: Model,
    critic_target: Model,
    batch: Batch,
    temp: float,
) -> Tuple[PRNGKey, ContinuousDDPM, Metric]:
    a0 = batch.action
    rng, at, t, eps = actor.add_noise(rng, a0)
    alpha, sigma = actor.noise_schedule_func(t)

    def get_q_value(at: jnp.ndarray, obs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        ft = nce_target(obs, at, t, method="forward_phi")
        q = critic_target(ft)
        return q.mean(axis=0).mean()
    q_grad_fn = jax.vmap(jax.grad(get_q_value))
    q_grad = q_grad_fn(at, batch.obs, t)
    eps_estimation = - sigma * q_grad / temp / (jnp.abs(q_grad).mean() + 1e-6)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": actor_params},
            at,
            t,
            condition=batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((eps_pred - eps_estimation) ** 2).mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/eps_estimation_l1": jnp.abs(eps_estimation).mean(),
        }
    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, actor_metrics


class ACAAgent(BaseAgent):
    """
    ACA (Actor-Critic with Actor) agent.
    """
    name = "ACAAgent"
    model_names = ["nce", "nce_target", "actor", "actor_target", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: ACAConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg

        self.feature_dim = cfg.feature_dim
        self.ranking = cfg.ranking
        self.linear = cfg.linear
        self.reward_coef = cfg.reward_coef
        self.critic_coef = cfg.critic_coef

        self.rng, nce_rng, nce_init_rng, actor_rng, critic_rng = jax.random.split(self.rng, 5)

        # define the nce
        nce_def = FactorizedNCE(
            self.obs_dim,
            self.act_dim,
            self.feature_dim,
            cfg.phi_hidden_dims,
            cfg.mu_hidden_dims,
            cfg.reward_hidden_dims,
            cfg.rff_dim,
            cfg.num_noises,
            self.ranking,
        )
        self.nce = Model.create(
            nce_def,
            nce_rng,
            inputs=(
                nce_init_rng,
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
                nce_init_rng,
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
        )

        # define the actor
        time_embedding = partial(LearnableFourierEmbedding, output_dim=cfg.diffusion.time_dim)
        cond_embedding = partial(MLP, hidden_dims=(128, 128), activation=mish)
        noise_predictor = partial(
            MLP,
            hidden_dims=cfg.diffusion.mlp_hidden_dims,
            output_dim=act_dim,
            activation=mish,
            layer_norm=False,
            dropout=None,
        )
        backbone_def = ContinuousDDPMBackbone(
            noise_predictor=noise_predictor,
            time_embedding=time_embedding,
            cond_embedding=cond_embedding,
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

        self.actor = ContinuousDDPM.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule="cosine",
            noise_schedule_params={},
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            t_schedule_n=1.0,
            optimizer=optax.adam(learning_rate=actor_lr),
        )

        # define the critic
        critic_activation = {
            "relu": jax.nn.relu,
            "elu": jax.nn.elu,
        }[cfg.critic_activation]
        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic_hidden_dims,
            activation=critic_activation,
            layer_norm=True,
            dropout=None,
            ensemble_size=2,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim))),
        )

        # define tracking variables
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        metrics = {}

        self.rng, self.nce, nce_metrics = update_factorized_nce(
            self.rng,
            self.nce,
            batch,
            self.ranking,
            self.reward_coef,
        )
        metrics.update(nce_metrics)
        self.rng, self.critic, critic_metrics = jit_update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor,
            self.nce_target,
            batch,
            discount=self.cfg.discount,
            solver=self.cfg.diffusion.solver,
            critic_coef=self.critic_coef,
        )
        metrics.update(critic_metrics)
        self.rng, self.actor, actor_metrics = jit_update_actor(
            self.rng,
            self.actor,
            self.nce_target,
            self.critic_target,
            batch,
            temp=self.cfg.temp,
        )
        metrics.update(actor_metrics)

        if self._n_training_steps % self.cfg.target_update_freq == 0:
            self.sync_target()

        self._n_training_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        # if deterministic is true, sample cfg.num_samples actions and select the best one
        # if not, sample 1 action
        if deterministic:
            num_samples = self.cfg.num_samples
        else:
            num_samples = 1
        self.rng, action = jit_sample_actions(
            self.rng,
            self.actor,
            self.critic,
            self.nce_target,
            obs,
            training=False,
            num_samples=num_samples,
            solver=self.cfg.diffusion.solver,
        )
        if not deterministic:
            action = action + 0.1 * jax.random.normal(self.rng, action.shape)
        return action, {}

    def sync_target(self):
        self.critic_target = ema_update(self.critic, self.critic_target, self.cfg.ema)
        self.nce_target = ema_update(self.nce, self.nce_target, self.cfg.feature_ema)
