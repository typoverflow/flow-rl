from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.qsm import QSMConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import Ensemblize, ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "num_samples", "solver"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic: Model,
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
        qs = critic(obs_repeat, actions)
        qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions

@partial(jax.jit, static_argnames=("discount", "solver", "ema"))
def jit_update_qsm_critic(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic: Model,
    critic_target: Model,
    batch: Batch,
    discount: float,
    solver: str,
    ema: float,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    rng, next_xT_rng = jax.random.split(rng)
    next_xT = jax.random.normal(next_xT_rng, (*batch.next_obs.shape[:-1], actor.x_dim))
    rng, next_action, _ = actor.sample(rng, next_xT, batch.next_obs, training=False, solver=solver)
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
            "misc/next_action_l1": jnp.abs(next_action).mean(),
        }

    new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

    new_critic_target = ema_update(new_critic, critic_target, ema)
    return rng, new_critic, new_critic_target, critic_metrics

@partial(jax.jit, static_argnames=("temp",))
def jit_update_qsm_actor(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic_target: Model,
    batch: Batch,
    temp: float,
) -> Tuple[PRNGKey, ContinuousDDPM, Metric]:
    a0 = batch.action
    rng, at, t, eps = actor.add_noise(rng, a0)
    alpha1, alpha2 = actor.noise_schedule_func(t)

    q_grad_fn = jax.vmap(jax.grad(lambda a, s: critic_target(s, a).mean()))
    q_grad = q_grad_fn(at, batch.obs)
    eps_estimation = - alpha2 * q_grad / temp / (jnp.abs(q_grad).mean() + 1e-6)

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
            "misc/eps_estimation_std": jnp.std(eps_estimation, axis=0).mean(),
        }

    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, actor_metrics


class QSMAgent(BaseAgent):
    """
    Learning a Diffusion Model Policy from Rewards via Q-Score Matching (QSM)
    https://arxiv.org/abs/2312.11752
    """
    name = "QSMAgent"
    model_names = ["actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: QSMConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        # define the actor
        backbone_def = ContinuousDDPMBackbone(
            noise_predictor=MLP(
                hidden_dims=cfg.diffusion.mlp_hidden_dims,
                output_dim=act_dim,
                activation=mish,
                layer_norm=False,
                dropout=None,
            ),
            time_embedding=LearnableFourierEmbedding(
                output_dim=cfg.diffusion.time_dim
            ),
            cond_embedding=MLP(
                hidden_dims=(128, 128),
                activation=mish
            ),
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
            "mish": mish,
        }[cfg.critic_activation]
        critic_def = Ensemblize(
            base=ScalarCritic(
                backbone=MLP(
                    hidden_dims=cfg.critic_hidden_dims,
                    activation=critic_activation,
                    layer_norm=False,
                    dropout=None,
                ),
            ),
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

        # define tracking variables
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.critic, self.critic_target, critic_metrics = jit_update_qsm_critic(
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            batch,
            discount=self.cfg.discount,
            solver=self.cfg.diffusion.solver,
            ema=self.cfg.ema,
        )
        self.rng, self.actor, actor_metrics = jit_update_qsm_actor(
            self.rng,
            self.actor,
            self.critic_target,
            batch,
            temp=self.cfg.temp,
        )
        self._n_training_steps += 1
        return {**critic_metrics, **actor_metrics}

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
            obs,
            training=False,
            num_samples=num_samples,
            solver=self.cfg.diffusion.solver,
        )
        if not deterministic:
            action = action + 0.1 * jax.random.normal(self.rng, action.shape)
        return action, {}
