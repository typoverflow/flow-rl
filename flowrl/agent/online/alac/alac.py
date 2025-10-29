from functools import partial
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.alac.network import EnsembleEnergyNet
from flowrl.config.online.mujoco.algo.alac import ALACConfig
from flowrl.flow.langevin_dynamics import AnnealedLangevinDynamics
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.mlp import MLP, ResidualMLP
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "num_samples"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: AnnealedLangevinDynamics,
    ld: AnnealedLangevinDynamics,
    obs,
    training: bool,
    num_samples: int
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, x_init_rng = jax.random.split(rng)
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    x_init = jax.random.normal(x_init_rng, (*obs_repeat.shape[:-1], actor.x_dim))
    rng, actions, _ = actor.sample(rng, x_init, obs_repeat, training=training)
    if num_samples == 1:
        actions = actions[:, 0]
    else:
        qs = ld(actions, t=jnp.zeros((B, num_samples, 1), dtype=jnp.float32), condition=obs_repeat)
        qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions

@partial(jax.jit, static_argnames=("discount", "ema"))
def jit_update_ld(
    rng: PRNGKey,
    ld: AnnealedLangevinDynamics,
    ld_target: AnnealedLangevinDynamics,
    actor: AnnealedLangevinDynamics,
    batch: Batch,
    discount: float,
    ema: float,
) -> Tuple[PRNGKey, AnnealedLangevinDynamics, AnnealedLangevinDynamics, Metric]:
    B, A = batch.action.shape[0], batch.action.shape[1]
    feed_t = jnp.zeros((B, 1), dtype=jnp.float32)

    rng, next_xT_rng = jax.random.split(rng)
    # next_action_init = jax.random.normal(next_xT_rng, (*batch.next_obs.shape[:-1], ld.x_dim))
    next_action_init = jax.random.normal(next_xT_rng, (*batch.next_obs.shape[:-1], actor.x_dim))
    # rng, next_action, history = ld_target.sample(
    #     rng,
    #     next_action_init,
    #     batch.next_obs,
    #     training=False,
    # )
    rng, next_action, history = actor.sample(
        rng,
        next_action_init,
        batch.next_obs,
        training=False,
    )
    q_target = ld_target(next_action, feed_t, batch.next_obs, training=False)
    # q_target = ld_target(batch.next_obs, next_action, training=False)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target.min(axis=0)

    def ld_loss_fn(params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q_pred = ld.apply(
            {"params": params},
            batch.action,
            t=feed_t,
            condition=batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        # q_pred = ld.apply(
        #     {"params": params},
        #     batch.obs,
        #     batch.action,
        #     training=True,
        #     rngs={"dropout": dropout_rng},
        # )
        ld_loss = ((q_pred - q_target[jnp.newaxis, :])**2).mean()
        return ld_loss, {
            "loss/ld_loss": ld_loss,
            "misc/q_mean": q_pred.mean(),
            "misc/reward": batch.reward.mean(),
            # "misc/q_grad_l1": jnp.abs(history[1]).mean(),
        }

    new_ld, ld_metrics = ld.apply_gradient(ld_loss_fn)
    new_ld_target = ema_update(new_ld, ld_target, ema)

    # record energy
    # num_checkpoints = 5
    # stepsize_checkpoint = ld.steps // num_checkpoints
    # energy_history = history[2][jnp.arange(0, ld.steps, stepsize_checkpoint)]
    # energy_history = energy_history.mean(axis=[-2, -1])
    # ld_metrics.update({
    #     f"info/energy_step{i}": energy for i, energy in enumerate(energy_history)
    # })

    return rng, new_ld, new_ld_target, ld_metrics


@partial(jax.jit, static_argnames=())
def jit_update_actor(
    rng: PRNGKey,
    actor: AnnealedLangevinDynamics,
    critic_target: AnnealedLangevinDynamics,
    batch: Batch,
) -> Tuple[PRNGKey, AnnealedLangevinDynamics, Metric]:
    x0 = batch.action
    rng, xt, t, eps = actor.add_noise(rng, x0)
    # rng, t_rng, noise_rng = jax.random.split(rng, 3)
    # t = jax.random.uniform(t_rng, (*x0.shape[:-1], 1), dtype=jnp.float32, minval=actor.t_diffusion[0], maxval=actor.t_diffusion[1])
    # eps = jax.random.normal(noise_rng, x0.shape, dtype=jnp.float32)
    alpha, sigma = actor.noise_schedule_func(t)
    xt = alpha * x0 + sigma * eps

    q_grad_fn = jax.vmap(jax.grad(lambda a, s: critic_target(a, None, condition=s).min(axis=0).mean()))
    # q_grad_fn = jax.vmap(jax.grad(lambda a, s: critic_target(s, a).min(axis=0).mean()))
    q_grad = q_grad_fn(xt, batch.obs)
    q_grad = alpha * q_grad - sigma * xt
    eps_estimation = sigma * q_grad / (jnp.abs(q_grad).mean() + 1e-6)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": actor_params},
            xt,
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


class ALACAgent(BaseAgent):
    """
    Annealed Langevin Dynamics Actor-Critic (ALAC) agent.
    """
    name = "ALACAgent"
    model_names = ["actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: ALACConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, ld_rng = jax.random.split(self.rng, 2)

        # define the critic
        # from flowrl.module.critic import EnsembleCritic
        # from flowrl.module.model import Model
        # critic_def = EnsembleCritic(
        #     hidden_dims=cfg.ld.hidden_dims,
        #     activation=jax.nn.relu,
        #     layer_norm=False,
        #     dropout=None,
        #     ensemble_size=2,
        # )
        # self.ld = Model.create(
        #     critic_def,
        #     ld_rng,
        #     inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        #     optimizer=optax.adam(learning_rate=cfg.ld.lr),
        # )
        # self.ld_target = Model.create(
        #     critic_def,
        #     ld_rng,
        #     inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        # )

        mlp_impl = ResidualMLP if cfg.ld.resnet else MLP
        activation = {"mish": mish, "relu": jax.nn.relu}[cfg.ld.activation]
        energy_def = EnsembleEnergyNet(
            mlp_impl=mlp_impl,
            hidden_dims=cfg.ld.hidden_dims,
            output_dim=1,
            activation=activation,
            layer_norm=False,
            dropout=None,
            ensemble_size=cfg.ld.ensemble_size,
            # time_embedding=partial(LearnableFourierEmbedding, output_dim=cfg.ld.time_dim),
            time_embedding=None,
            # cond_embedding=partial(MLP, hidden_dims=cfg.ld.cond_hidden_dims, activation=activation),
            cond_embedding=None,
        )
        self.ld = AnnealedLangevinDynamics.create(
            network=energy_def,
            rng=ld_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.ones((1, 1)), jnp.ones((1, self.obs_dim))),
            x_dim=self.act_dim,
            grad_prediction=False,
            steps=cfg.ld.steps,
            step_size=cfg.ld.step_size,
            noise_scale=cfg.ld.noise_scale,
            noise_schedule=cfg.ld.noise_schedule,
            noise_schedule_params={},
            clip_sampler=cfg.ld.clip_sampler,
            x_min=cfg.ld.x_min,
            x_max=cfg.ld.x_max,
            t_schedule_n=1.0,
            epsilon=cfg.ld.epsilon,
            optimizer=optax.adam(learning_rate=cfg.ld.lr),
            clip_grad_norm=cfg.ld.clip_grad_norm,
        )
        self.ld_target = AnnealedLangevinDynamics.create(
            network=energy_def,
            rng=ld_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.ones((1, 1)), jnp.ones((1, self.obs_dim))),
            x_dim=self.act_dim,
            grad_prediction=False,
            steps=cfg.ld.steps,
            step_size=cfg.ld.step_size,
            noise_scale=cfg.ld.noise_scale,
            noise_schedule=cfg.ld.noise_schedule,
            noise_schedule_params={},
            clip_sampler=cfg.ld.clip_sampler,
            x_min=cfg.ld.x_min,
            x_max=cfg.ld.x_max,
            t_schedule_n=1.0,
            epsilon=cfg.ld.epsilon,
        )

        # DEBUG define the actor
        from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
        self.rng, actor_rng = jax.random.split(self.rng, 2)
        time_embedding = partial[LearnableFourierEmbedding](LearnableFourierEmbedding, output_dim=cfg.ld.time_dim)
        cond_embedding = partial(MLP, hidden_dims=[128, 128], activation=mish)
        noise_predictor = partial(
            MLP,
            hidden_dims=cfg.ld.hidden_dims,
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
        # self.actor = ContinuousDDPM.create(
        #     network=backbone_def,
        #     rng=actor_rng,
        #     inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
        #     x_dim=self.act_dim,
        #     steps=cfg.ld.steps,
        #     noise_schedule="cosine",
        #     noise_schedule_params={},
        #     clip_sampler=cfg.ld.clip_sampler,
        #     x_min=cfg.ld.x_min,
        #     x_max=cfg.ld.x_max,
        #     t_schedule_n=1.0,
        #     optimizer=optax.adam(learning_rate=cfg.ld.lr),
        # )
        self.actor = AnnealedLangevinDynamics.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.ones((1, 1)), jnp.ones((1, self.obs_dim))),
            x_dim=self.act_dim,
            grad_prediction=True,
            steps=cfg.ld.steps,
            step_size=cfg.ld.step_size,
            noise_scale=cfg.ld.noise_scale,
            noise_schedule="cosine",
            noise_schedule_params={},
            clip_sampler=cfg.ld.clip_sampler,
            x_min=cfg.ld.x_min,
            x_max=cfg.ld.x_max,
            t_schedule_n=1.0,
            epsilon=cfg.ld.epsilon,
            optimizer=optax.adam(learning_rate=cfg.ld.lr),
            clip_grad_norm=cfg.ld.clip_grad_norm,
        )

        # define tracking variables
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.ld, self.ld_target, ld_metrics = jit_update_ld(
            self.rng,
            self.ld,
            self.ld_target,
            self.actor,
            batch,
            self.cfg.discount,
            self.cfg.ema,
        )
        self.rng, self.actor, actor_metrics = jit_update_actor(
            self.rng,
            self.actor,
            self.ld_target,
            batch,
        )

        self._n_training_steps += 1
        return {**ld_metrics, **actor_metrics}

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
            # self.ld,
            self.actor,
            self.ld,
            obs,
            training=False,
            num_samples=num_samples,
        )
        if not deterministic:
            action = action + 0.1 * jax.random.normal(self.rng, action.shape)
        return action, {}
