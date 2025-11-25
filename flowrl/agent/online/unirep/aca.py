from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.unirep.network import FactorizedNCE, update_factorized_nce
from flowrl.config.online.mujoco.algo.unirep.aca import ACAConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
from flowrl.flow.ddpm import DDPM, DDPMBackbone
from flowrl.functional.activation import l2_normalize, mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import Critic, EnsembleCritic, EnsembleCriticT
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.rff import RffEnsembleCritic
from flowrl.module.time_embedding import LearnableFourierEmbedding, PositionalEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "num_samples", "solver"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: Model,
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
        # t0 = jnp.zeros((obs_repeat.shape[0], num_samples, 1))
        # f0 = nce_target(obs_repeat, actions, t0, method="forward_phi")
        # qs = critic(obs_repeat, actions)
        t1 = jnp.ones((obs_repeat.shape[0], num_samples, 1), dtype=jnp.int32)
        f1 = nce_target(obs_repeat, actions, t1, method="forward_phi")
        qs = critic(f1).min(axis=0).reshape(B, num_samples)
        # qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions

@partial(jax.jit, static_argnames=("deterministic", "exploration_noise"))
def jit_td3_sample_action(
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

@partial(jax.jit, static_argnames=("discount", "solver", "critic_coef"))
def jit_update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    value: Model,
    value_target: Model,
    actor: Model,
    backup: Model,
    nce_target: Model,
    batch: Batch,
    discount: float,
    solver: str,
    critic_coef: float,
) -> Tuple[PRNGKey, Model, Metric]:
    # q0 target
    B = batch.obs.shape[0]
    A = batch.action.shape[-1]
    t1 = jnp.ones((batch.obs.shape[0], 1), dtype=jnp.int32)
    a0 = batch.action


    rng, next_aT_rng = jax.random.split(rng)
    next_a0 = backup(batch.next_obs, training=False)
    # next_aT = jax.random.normal(next_aT_rng, (*batch.next_obs.shape[:-1], actor.x_dim))
    # rng, next_a0, _ = actor.sample(rng, next_aT, batch.next_obs, training=False, solver=solver)
    # next_f0 = nce_target(batch.next_obs, next_a0, t0, method="forward_phi")
    # q0_target = critic_target(next_f0)
    next_f1 = nce_target(batch.next_obs, next_a0, t1, method="forward_phi")
    f1 = nce_target(batch.obs, a0, t1, method="forward_phi")
    # q0_target = critic_target(batch.next_obs, next_a0)
    q0_target = critic_target(next_f1)
    q0_target = batch.reward + discount * (1 - batch.terminal) * q0_target.min(axis=0)


    # features
    # rng, at, t, eps = actor.add_noise(rng, a0)
    # weight_t = actor.alpha_hats[t] / (1-actor.alpha_hats[t])
    # weight_t = 1.0
    # ft = nce_target(batch.obs, at, t, method="forward_phi")
    # rng, t_rng, noise_rng = jax.random.split(rng, 3)
    # t = jax.random.randint(t_rng, (*a0.shape[:-1], 1), 0, actor.steps+1)
    # t = jnp.ones((*a0.shape[:-1], 1), dtype=jnp.int32)
    # eps = jax.random.normal(noise_rng, a0.shape)
    # at = jnp.sqrt(actor.alpha_hats[t]) * a0 + jnp.sqrt(1 - actor.alpha_hats[t]) * eps

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        # f0 = nce_target(batch.obs, a0, t0, method="forward_phi")
        q0_pred = critic.apply(
            {"params": critic_params},
            f1,
            # batch.obs,
            # a0,
        )
        critic_loss = (
            ((q0_pred - q0_target[jnp.newaxis, :])**2).mean()
        )
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q0_mean": q0_pred.mean(),
            "misc/reward": batch.reward.mean(),
            "misc/next_action_l1": jnp.abs(next_a0).mean(),
            "misc/q0_target": q0_target.mean(),
        }

    new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

    # rng, at, t, eps = actor.add_noise(rng, a0)
    rng, t_rng, noise_rng = jax.random.split(rng, 3)
    t1 = jnp.ones((batch.obs.shape[0], 1), dtype=jnp.int32)
    eps = jax.random.normal(noise_rng, a0.shape)
    at = jnp.sqrt(actor.alpha_hats[t1]) * a0 + jnp.sqrt(1 - actor.alpha_hats[t1]) * eps
    ft = nce_target(batch.obs, at, t1, method="forward_phi")

    def value_loss_fn(value_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        qt_pred = value.apply(
            {"params": value_params},
            ft,
        )
        value_loss = ((qt_pred - q0_target)**2).mean()
        return value_loss, {
            "loss/value_loss": value_loss,
            "misc/qt_mean": qt_pred.mean(),
        }
    new_value, value_metrics = value.apply_gradient(value_loss_fn)

    return rng, new_critic, new_value, {
        **critic_metrics,
        **value_metrics,
    }

def jit_compute_metrics(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    value: Model,
    nce_target: Model,
    batch: Batch,
) -> Tuple[PRNGKey, Metric]:
    B, S = batch.obs.shape
    A = batch.action.shape[-1]
    num_actions = 50
    metrics = {}
    rng, action_rng = jax.random.split(rng)
    obs_repeat = batch.obs[..., jnp.newaxis, :].repeat(num_actions, axis=-2)
    action_repeat = batch.action[..., jnp.newaxis, :].repeat(num_actions, axis=-2)
    action_repeat = jax.random.uniform(action_rng, (B, num_actions, A), minval=-1.0, maxval=1.0)

    def get_critic(at, obs):
        t1 = jnp.ones((1, ), dtype=jnp.int32)
        f1 = nce_target(obs, at, t1, method="forward_phi")
        q = critic(f1)
        return q.mean()
    all_critic, all_critic_grad = jax.vmap(jax.value_and_grad(get_critic))(
        action_repeat.reshape(-1, A),
        obs_repeat.reshape(-1, S),
    )
    all_critic = all_critic.reshape(B, num_actions, 1)
    all_critic_grad = all_critic_grad.reshape(B, num_actions, -1)
    metrics.update({
        f"q_std/critic": all_critic.std(axis=1).mean(),
        f"q_grad/critic": jnp.abs(all_critic_grad).mean(),
    })

    def get_value(at, obs, t):
        ft = nce_target(obs, at, t, method="forward_phi")
        q = value(ft)
        return q.mean()

    for t in [0] + list(range(1, actor.steps+1, actor.steps//5)):
        t_input = jnp.ones((B, num_actions, 1)) * t
        all_value, all_value_grad = jax.vmap(jax.value_and_grad(get_value))(
            action_repeat.reshape(-1, A),
            obs_repeat.reshape(-1, S),
            t_input.reshape(-1, 1),
        )
        all_value = all_value.reshape(B, num_actions, 1)
        all_value_grad = all_value_grad.reshape(B, num_actions, -1)
        metrics.update({
            f"q_std/value_{t}": all_value.std(axis=1).mean(),
            f"q_grad/value_{t}": jnp.abs(all_value_grad).mean(),
        })
    return rng, metrics


@partial(jax.jit, static_argnames=("temp",))
def jit_update_actor(
    rng: PRNGKey,
    actor: Model,
    backup: Model,
    nce_target: Model,
    critic_target: Model,
    value_target: Model,
    batch: Batch,
    temp: float,
) -> Tuple[PRNGKey, Model, Metric]:
    a0 = batch.action
    t1 = jnp.ones((batch.obs.shape[0], 1), dtype=jnp.int32)
    rng, at, t, eps = actor.add_noise(rng, a0)
    sigma = jnp.sqrt(1 - actor.alpha_hats[t])
    def get_q_value(at: jnp.ndarray, obs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        t1 = jnp.ones(t.shape, dtype=jnp.int32)
        ft = nce_target(obs, at, t1, method="forward_phi")
        q = value_target(ft)
        return q.mean()
    q_grad_fn = jax.vmap(jax.grad(get_q_value))
    q_grad = q_grad_fn(at, batch.obs, t)
    q_grad_l1 = jnp.abs(q_grad).mean()
    eps_estimation = - sigma * q_grad / temp / (jnp.abs(q_grad).mean() + 1e-6)
    # eps_estimation = -sigma * l2_normalize(q_grad) / temp

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
            "misc/q_grad_l1": q_grad_l1,
        }
    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

    def backup_loss_fn(backup_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        new_action = backup.apply(
            {"params": backup_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        f1 = nce_target(batch.obs, new_action, t1, method="forward_phi")
        q = critic_target(f1)
        loss = - q.mean()
        return loss, {
            "loss/backup_loss": loss,
        }
    new_backup, backup_metrics = backup.apply_gradient(backup_loss_fn)

    return rng, new_actor, new_backup, {
        **actor_metrics,
        **backup_metrics,
    }


class ACAAgent(BaseAgent):
    """
    ACA (Actor-Critic with Actor) agent.
    """
    name = "ACAAgent"
    model_names = ["nce", "nce_target", "actor", "critic", "critic_target", "value", "value_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: ACAConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg

        self.feature_dim = cfg.feature_dim
        self.ranking = cfg.ranking
        self.linear = cfg.linear
        self.reward_coef = cfg.reward_coef
        self.critic_coef = cfg.critic_coef

        self.rng, nce_rng, nce_init_rng, actor_rng, critic_rng, value_rng = jax.random.split(self.rng, 6)

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
        time_embedding = partial(PositionalEmbedding, output_dim=cfg.diffusion.time_dim)
        cond_embedding = partial(MLP, hidden_dims=(128, 128), activation=mish)
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

        self.actor = DDPM.create(
            network=backbone_def,
            rng=actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule="cosine",
            noise_schedule_params={},
            approx_postvar=False,
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            optimizer=optax.adam(learning_rate=actor_lr),
        )

        # define the backup actor
        from flowrl.module.actor import SquashedDeterministicActor
        backup_def = SquashedDeterministicActor(
            backbone=MLP(
                hidden_dims=[512,512,512],
                layer_norm=True,
                dropout=None,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
        )
        self.backup = Model.create(
            backup_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=actor_lr),
        )

        # define the critic
        critic_activation = {
            "relu": jax.nn.relu,
            "elu": jax.nn.elu,
        }[cfg.critic_activation]
        # critic_def = EnsembleCritic(
        #     hidden_dims=[512, 512],
        #     activation=critic_activation,
        #     ensemble_size=2,
        #     layer_norm=True,
        # )
        critic_def = RffEnsembleCritic(
            feature_dim=self.feature_dim,
            hidden_dims=[512,],
            rff_dim=cfg.rff_dim,
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

        value_def = Critic(
            hidden_dims=[512, 512],
            activation=critic_activation,
            layer_norm=True,
        )
        self.value = Model.create(
            value_def,
            value_rng,
            inputs=(jnp.ones((1, self.feature_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.value_target = Model.create(
            value_def,
            value_rng,
            inputs=(jnp.ones((1, self.feature_dim))),
        )

        # from flowrl.agent.online.unirep.network import (
        #     EnsembleACACritic,
        #     EnsembleResidualCritic,
        #     ResidualCritic,
        #     SeparateCritic,
        # )

        # value_def = EnsembleACACritic(
        #     time_dim=16,
        #     hidden_dims=[256,256,256],
        #     activation=jax.nn.mish,
        #     ensemble_size=2,
        # )
        # value_def = EnsembleResidualCritic(
        #     time_embedding=time_embedding,
        #     hidden_dims=[512, 512, 512],
        #     activation=jax.nn.mish,
        # )
        # value_def = SeparateCritic(
        #     hidden_dims=[512, 512, 512],
        #     activation=jax.nn.mish,
        #     ensemble_size=cfg.diffusion.steps+1,
        # )
        # self.value = Model.create(
        #     value_def,
        #     value_rng,
        #     inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
        #     optimizer=optax.adam(learning_rate=cfg.critic_lr),
        # )
        # self.value_target = Model.create(
        #     value_def,
        #     value_rng,
        #     inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
        # )
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
        self.rng, self.critic, self.value, critic_metrics = jit_update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.value,
            self.value_target,
            self.actor,
            self.backup,
            self.nce_target,
            batch,
            discount=self.cfg.discount,
            solver=self.cfg.diffusion.solver,
            critic_coef=self.critic_coef,
        )
        metrics.update(critic_metrics)
        self.rng, self.actor, self.backup, actor_metrics = jit_update_actor(
            self.rng,
            self.actor,
            self.backup,
            self.nce_target,
            self.critic_target,
            self.value_target,
            batch,
            temp=self.cfg.temp,
        )
        metrics.update(actor_metrics)

        if self._n_training_steps % self.cfg.target_update_freq == 0:
            self.sync_target()

        if self._n_training_steps % 2000 == 0:
            self.rng, metrics = jit_compute_metrics(
                self.rng,
                self.actor,
                self.critic,
                self.value,
                self.nce_target,
                batch,
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
            num_samples = self.cfg.num_samples
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
        else:
            self.rng, action_rng = jax.random.split(self.rng)
            action = jit_td3_sample_action(
                action_rng,
                self.backup,
                obs,
                deterministic,
                exploration_noise=0.2,
            )
        return action, {}

    def sync_target(self):
        self.critic_target = ema_update(self.critic, self.critic_target, self.cfg.ema)
        self.value_target = ema_update(self.value, self.value_target, self.cfg.ema)
        self.nce_target = ema_update(self.nce, self.nce_target, self.cfg.feature_ema)
