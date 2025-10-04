from functools import partial
from operator import attrgetter
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.online.ctrl.network import FactorizedNCE, update_factorized_nce
from flowrl.agent.online.td3 import TD3Agent
from flowrl.config.online.mujoco.algo.ctrl_td3 import CTRL_TD3_Config
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedDeterministicActor
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.rff import RffEnsembleCritic
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("discount", "target_policy_noise", "noise_clip"))
def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_target: Model,
    nce: Model,
    nce_target: Model,
    batch: Batch,
    discount: float,
    target_policy_noise: float,
    noise_clip: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, batch.action.shape) * target_policy_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_action = jnp.clip(actor_target(batch.next_obs) + noise, -1.0, 1.0)

    next_feature = nce_target(batch.next_obs, next_action, method="forward_phi")
    q_target = critic_target(next_feature).min(0)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target

    back_critic_grad = False
    if back_critic_grad:
        # this part will use feature
        raise NotImplementedError("no back critic grad exists")

    feature = nce_target(batch.obs, batch.action, method="forward_phi")

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q_pred = critic.apply(
            {"params": critic_params},
            feature,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((q_pred - q_target[jnp.newaxis, :])**2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q_pred.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics


@jax.jit
def update_actor(
    rng: PRNGKey,
    actor: Model,
    nce: Model,
    nce_target: Model,
    critic: Model,
    batch: Batch,
) -> Tuple[PRNGKey, Model, Metric]:
    def actor_loss_fn(
        actor_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        new_action = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        new_feature = nce_target(batch.obs, new_action, method="forward_phi")
        q = critic(new_feature)
        actor_loss = - q.mean()

        return actor_loss, {
            "loss/actor_loss": actor_loss,
        }

    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, metrics


class Ctrl_TD3_Agent(TD3Agent):
    """
    CTRL Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.
    """

    name = "CTRLTD3Agent"
    model_names = ["nce", "actor", "actor_target", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: CTRL_TD3_Config, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg

        self.ctrl_coef = cfg.ctrl_coef
        self.critic_coef = cfg.critic_coef

        self.aug_batch_size = cfg.aug_batch_size
        self.feature_tau = cfg.feature_tau
        self.linear = cfg.linear
        self.ranking = cfg.ranking
        self.feature_dim = cfg.feature_dim
        self.num_noises = cfg.num_noises
        self.reward_coef = cfg.reward_coef
        self.rff_dim = cfg.rff_dim

        # sanity checks for the hyper-parameters
        assert not self.linear, "Removing linear version for now"

        # networks
        self.rng, nce_rng, actor_rng, critic_rng = jax.random.split(self.rng, 4)
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
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
        )

        actor_def = SquashedDeterministicActor(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.layer_norm,
                dropout=None,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
        )
        critic_def = RffEnsembleCritic(
            feature_dim=self.feature_dim,
            hidden_dims=cfg.critic_hidden_dims,
            rff_dim=cfg.rff_dim,
            ensemble_size=2,
        )
        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.actor_target = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
        )

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        metrics = {}

        split_index = batch.obs.shape[0] - self.aug_batch_size
        obs, action, next_obs, reward, terminal = [
            b[:split_index]
            for b in attrgetter("obs", "action", "next_obs", "reward", "terminal")(
                batch
            )
        ]
        fobs, faction, fnext_obs, freward, fterminal = [
            b[split_index:]
            for b in attrgetter("obs", "action", "next_obs", "reward", "terminal")(
                batch
            )
        ]
        rl_batch = Batch(obs, action, reward, terminal, next_obs, None)

        self.rng, self.nce, nce_metrics = update_factorized_nce(
            self.rng,
            self.nce,
            fobs,
            faction,
            fnext_obs,
            freward,
            self.ranking,
            self.reward_coef,
        )
        metrics.update(nce_metrics)

        self.rng, self.critic, critic_metrics = update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor_target,
            self.nce,
            self.nce_target,
            rl_batch,
            discount=self.cfg.discount,
            target_policy_noise=self.target_policy_noise,
            noise_clip=self.noise_clip,
        )
        metrics.update(critic_metrics)

        if self._n_training_steps % self.actor_update_freq == 0:
            self.rng, self.actor, actor_metrics = update_actor(
                self.rng,
                self.actor,
                self.nce,
                self.nce_target,
                self.critic,
                rl_batch,
            )
            metrics.update(actor_metrics)

        if self._n_training_steps % self.target_update_freq == 0:
            self.sync_target()

        self._n_training_steps += 1
        return metrics

    def sync_target(self):
        super().sync_target()
        self.nce_target = ema_update(self.nce, self.nce_target, self.feature_tau)
