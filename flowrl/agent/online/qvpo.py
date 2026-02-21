from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.qvpo import QVPOConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "num_samples", "best_of_n"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    best_of_n: bool,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], actor.x_dim))
    rng, actions, _ = actor.sample(rng, xT, obs_repeat, training)
    if best_of_n:
        qs = critic(obs_repeat, actions)
        qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions

@partial(jax.jit, static_argnames=("discount", "ema", "reweight", "entropy_coef", "num_behavior_samples", "num_train_samples"))
def jit_update_qvpo(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    actor_target: ContinuousDDPM,
    critic: Model,
    critic_target: Model,
    running_q_mean: jnp.array,
    running_q_std: jnp.array,
    batch: Batch,
    discount: float,
    reweight: str,
    num_behavior_samples: int,
    num_train_samples: int,
    ema: float,
    entropy_coef: float,
) -> Tuple[PRNGKey, ContinuousDDPM, Model, Model, jnp.ndarray, jnp.ndarray, Metric]:

    # update critic
    rng, next_action = jit_sample_actions(
        rng,
        actor_target,
        critic_target,
        batch.next_obs,
        training=False,
        num_samples=num_behavior_samples,
        best_of_n=True,
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
    rng, action_batch = jit_sample_actions(
        rng,
        actor,
        critic,
        batch.obs,
        training=False,
        num_samples=num_train_samples,
        best_of_n=False,
    )
    q_batch = jax.vmap(
        critic,
        in_axes=(None, 1),
        out_axes=2,
    )(batch.obs, action_batch)
    obs_batch = batch.obs
    q_batch = q_batch.mean(axis=0).squeeze(-1)
    v_batch = q_batch.mean(axis=1, keepdims=True)

    q_mean = q_batch.mean()
    q_std = q_batch.std()
    running_q_mean += 0.001 * (q_mean - running_q_mean)
    running_q_std += 0.001 * (q_std - running_q_std)

    # q_idx: [B, 1], action_batch: [B, N, A]
    # To gather actions with max Q, use jnp.take_along_axis with axis=1, matching for all A
    q_idx = jnp.argmax(q_batch, axis=1, keepdims=True)  # [B, 1]
    q_batch = jnp.take_along_axis(q_batch, q_idx, axis=1)  # [B, 1]
    action_batch = jnp.take_along_axis(
        action_batch,
        jnp.expand_dims(q_idx, axis=-1),  # [B, 1, 1]
        axis=1
    ).squeeze(1)
    if reweight == "adv":
        weight = q_batch - v_batch # no need to clip since only selecting the best action

    if entropy_coef >= 0:
        rng, rand_rng = jax.random.split(rng)
        rand_states = batch.obs.repeat(10, axis=0)
        rand_actions = jax.random.uniform(rand_rng, (*rand_states.shape[:-1], actor.x_dim))
        rand_weight = weight.repeat(10, axis=0) * entropy_coef

        action_batch = jnp.concatenate([action_batch, rand_actions], axis=0)
        obs_batch = jnp.concatenate([obs_batch, rand_states], axis=0)
        weight = jnp.concatenate([weight, rand_weight], axis=0)

    rng, at, t, eps = actor.add_noise(rng, action_batch)

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = actor.apply(
            {"params": actor_params},
            at,
            t,
            condition=obs_batch,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = (weight * (eps_pred - eps) ** 2).mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/weights": q_batch.mean(),
            "misc/weights_std": q_batch.std(0).mean(),
            "misc/weights_max": q_batch.max(0).mean(),
            "misc/weights_min": q_batch.min(0).mean(),
            "misc/batch_action_std": action_batch.std(axis=1).mean(),
        }
    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

    new_actor_target = ema_update(new_actor, actor_target, ema)
    new_critic_target = ema_update(new_critic, critic_target, ema)

    return rng, new_actor, new_actor_target, new_critic, new_critic_target, running_q_mean, running_q_std, {
        **critic_metrics,
        **actor_metrics,
        "misc/running_q_mean": running_q_mean,
        "misc/running_q_std": running_q_std,
    }


class QVPOAgent(BaseAgent):
    """
    Diffusion Policy Mirror Descent (DPMD) agent.
    """
    name = "DPMDAgent"
    model_names = ["actor", "critic", "actor_target", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: QVPOConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)

        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

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
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.actor_target = ContinuousDDPM.create(
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
        )

        critic_activation = {
            "relu": jax.nn.relu,
            "elu": jax.nn.elu,
            "mish": mish,
        }[cfg.critic_activation]
        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic_hidden_dims,
            activation=critic_activation,
            layer_norm=False,
            dropout=None,
            ensemble_size=2,
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

        self.running_q_mean = jnp.ones((1, ), dtype=jnp.float32)
        self.running_q_std = jnp.ones((1, ), dtype=jnp.float32)

        # define tracking variables
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.actor_target, self.critic, self.critic_target, self.running_q_mean, self.running_q_std, metrics = jit_update_qvpo(
            self.rng,
            self.actor,
            self.actor_target,
            self.critic,
            self.critic_target,
            self.running_q_mean,
            self.running_q_std,
            batch,
            discount=self.cfg.discount,
            reweight=self.cfg.reweight,
            num_behavior_samples=self.cfg.num_behavior_samples,
            num_train_samples=self.cfg.num_train_samples,
            ema=self.cfg.ema,
            entropy_coef=self.cfg.entropy_coef,
        )
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
            num_samples = self.cfg.num_evaluate_samples
        else:
            num_samples = 1
        self.rng, action = jit_sample_actions(
            self.rng,
            self.actor,
            self.critic,
            obs,
            training=False,
            num_samples=num_samples,
            best_of_n=deterministic,
        )
        if num_samples == 1:
            action = action[:, 0]
        return action, {}
