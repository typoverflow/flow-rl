from functools import partial
from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

import flowrl.module.initialization as init
from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.flac import FLACConfig
from flowrl.flow.cnf import ContinuousNormalizingFlow, FlowBackbone
from flowrl.functional.activation import get_activation
from flowrl.functional.categorical import project_categorical
from flowrl.functional.ema import ema_update
from flowrl.module.critic import CategoricalCritic, Ensemblize, ScalarCritic
from flowrl.module.misc import TunableCoefficient
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import Batch, Initializer, Metric, Param, PRNGKey, Sequence


def get_initializer(name: str) -> Callable[[], Initializer]:
    return {
        "default": init.default_kernel_init,
        "xavier_uniform": nn.initializers.xavier_uniform,
    }[name]


class FLACCriticBackbone(nn.Module):
    """Value block used by the official HumanoidBench FLAC critic."""

    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    layer_norm: bool = True
    kernel_init: Callable[[], Initializer] = nn.initializers.xavier_uniform
    bias_init: Callable[[], Initializer] = init.default_bias_init

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        if len(self.hidden_dims) != 4:
            raise ValueError(
                "FLACCriticBackbone requires [hidden, expanded, expanded, hidden]"
            )

        x = nn.Dense(
            self.hidden_dims[0],
            kernel_init=self.kernel_init(),
            bias_init=self.bias_init(),
        )(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.hidden_dims[1],
            kernel_init=self.kernel_init(),
            bias_init=self.bias_init(),
        )(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = self.activation(x)
        x = nn.Dense(
            self.hidden_dims[2],
            kernel_init=self.kernel_init(),
            bias_init=self.bias_init(),
        )(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = self.activation(x)
        x = nn.Dense(
            self.hidden_dims[3],
            kernel_init=self.kernel_init(),
            bias_init=self.bias_init(),
        )(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        return x


@partial(jax.jit, static_argnames=("training",))
def jit_sample_actions(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    obs: jnp.ndarray,
    training: bool = False,
    params: Param | None = None,
) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray]:
    rng, initial_rng, flow_rng = jax.random.split(rng, 3)
    initial_action = jnp.clip(
        jax.random.normal(initial_rng, (*obs.shape[:-1], actor.x_dim)),
        -1.0,
        1.0,
    )
    _, raw_action, (_, velocities) = actor.sample(
        flow_rng,
        initial_action,
        condition=obs,
        training=training,
        params=params,
        solver="midpoint",
    )
    per_step_energy = 0.5 * jnp.sum(
        jnp.square(velocities),
        axis=-1,
        keepdims=True,
    )
    kinetic = per_step_energy.sum(axis=0) / actor.steps
    action = jnp.tanh(raw_action)
    return rng, action, kinetic


@partial(jax.jit, static_argnames=("discount", "reward_scale"))
def update_scalar_critic(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    batch: Batch,
    discount: float,
    reward_scale: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, next_action, next_kinetic = jit_sample_actions(
        rng,
        actor,
        batch.next_obs,
    )
    target_q = critic_target(batch.next_obs, next_action).min(axis=0)
    scaled_reward = batch.reward * reward_scale
    target_q = scaled_reward + discount * (1 - batch.terminal) * (
        target_q - jnp.exp(log_alpha()) * next_kinetic
    )
    target_q = jax.lax.stop_gradient(target_q)

    def critic_loss_fn(
        critic_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = jnp.square(q - target_q[jnp.newaxis, ...]).sum(axis=0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q.mean(),
            "misc/reward": batch.reward.mean(),
            "misc/scaled_reward": scaled_reward.mean(),
            "misc/target_kinetic": next_kinetic.mean(),
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics


@partial(
    jax.jit,
    static_argnames=("discount", "reward_scale", "v_min", "v_max"),
)
def update_categorical_critic(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    batch: Batch,
    atoms: jnp.ndarray,
    discount: float,
    reward_scale: float,
    v_min: float,
    v_max: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, next_action, next_kinetic = jit_sample_actions(
        rng,
        actor,
        batch.next_obs,
    )
    next_logits = critic_target(batch.next_obs, next_action)
    next_probabilities = jax.nn.softmax(next_logits, axis=-1)
    next_expected_q = jnp.sum(next_probabilities * atoms, axis=-1)
    lower_critic = jnp.argmin(next_expected_q, axis=0)
    gather_indices = jnp.broadcast_to(
        lower_critic[jnp.newaxis, ..., jnp.newaxis],
        (1, *lower_critic.shape, next_probabilities.shape[-1]),
    )
    selected_probabilities = jnp.take_along_axis(
        next_probabilities,
        gather_indices,
        axis=0,
    )[0]

    scaled_reward = batch.reward * reward_scale
    bootstrap = discount * (1 - batch.terminal)
    target_atoms = scaled_reward + bootstrap * (
        atoms[jnp.newaxis, :] - jnp.exp(log_alpha()) * next_kinetic
    )
    target_distribution = jax.lax.stop_gradient(
        project_categorical(selected_probabilities, target_atoms, v_min, v_max)
    )
    clipped_fraction = jnp.mean((target_atoms < v_min) | (target_atoms > v_max))

    def critic_loss_fn(
        critic_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        logits = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        cross_entropy = -jnp.sum(
            target_distribution[jnp.newaxis, ...]
            * jax.nn.log_softmax(logits, axis=-1),
            axis=-1,
        )
        critic_loss = cross_entropy.sum(axis=0).mean()
        q = jnp.sum(jax.nn.softmax(logits, axis=-1) * atoms, axis=-1)
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q.mean(),
            "misc/reward": batch.reward.mean(),
            "misc/scaled_reward": scaled_reward.mean(),
            "misc/target_kinetic": next_kinetic.mean(),
            "misc/c51_clipped_fraction": clipped_fraction,
            "misc/c51_mass": target_distribution.sum(axis=-1).mean(),
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics


@partial(jax.jit, static_argnames=("distributional_critic",))
def update_actor(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    critic: Model,
    log_alpha: Model,
    batch: Batch,
    atoms: jnp.ndarray,
    distributional_critic: bool,
) -> Tuple[PRNGKey, ContinuousNormalizingFlow, jnp.ndarray, Metric]:
    rng, actor_rng = jax.random.split(rng)

    def actor_loss_fn(
        actor_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        sample_rng = jax.random.fold_in(actor_rng, dropout_rng[0])
        _, action, kinetic = jit_sample_actions(
            sample_rng,
            actor,
            batch.obs,
            training=True,
            params=actor_params,
        )
        if distributional_critic:
            probabilities = jax.nn.softmax(
                critic(batch.obs, action),
                axis=-1,
            )
            q = jnp.sum(
                probabilities * atoms,
                axis=-1,
                keepdims=True,
            )
        else:
            q = critic(batch.obs, action)
        min_q = q.min(axis=0)
        alpha = jnp.exp(log_alpha())
        actor_loss = (-min_q + jax.lax.stop_gradient(alpha) * kinetic).mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss,
            "misc/actor_q": min_q.mean(),
            "misc/kinetic": kinetic.mean(),
            "misc/action_l1": jnp.abs(action).mean(),
        }

    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, metrics["misc/kinetic"], metrics


@jax.jit
def update_alpha(
    log_alpha: Model,
    kinetic: jnp.ndarray,
    target_kinetic: float,
) -> Tuple[Model, Metric]:
    kinetic = jax.lax.stop_gradient(kinetic)

    def alpha_loss_fn(
        log_alpha_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        log_alpha_value = log_alpha.apply({"params": log_alpha_params})
        alpha_loss = log_alpha_value * (target_kinetic - kinetic)
        return alpha_loss, {
            "loss/alpha_loss": alpha_loss,
            "misc/alpha": jnp.exp(log_alpha_value),
        }

    return log_alpha.apply_gradient(alpha_loss_fn)


class FLACAgent(BaseAgent):
    """Field Least-Energy Actor-Critic (FLAC)."""

    name = "FLACAgent"
    model_names = ["actor", "critic", "critic_target", "log_alpha"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: FLACConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        if cfg.critic_ensemble_size < 2:
            raise ValueError("FLAC requires at least two critics")
        if cfg.critic_num_atoms < 2:
            raise ValueError("FLAC critic_num_atoms must be at least two")
        if cfg.critic_v_min >= cfg.critic_v_max:
            raise ValueError("FLAC critic_v_min must be smaller than critic_v_max")
        self.rng, actor_rng, critic_rng, alpha_rng = jax.random.split(self.rng, 4)

        actor_initializer = get_initializer(cfg.flow.initializer)
        critic_initializer = get_initializer(cfg.critic_initializer)
        actor_def = FlowBackbone(
            vel_predictor=MLP(
                hidden_dims=cfg.flow.hidden_dims,
                output_dim=act_dim,
                activation=get_activation(cfg.flow.activation),
                layer_norm=cfg.flow.layer_norm,
                kernel_init=actor_initializer,
            )
        )
        self.actor = ContinuousNormalizingFlow.create(
            network=actor_def,
            rng=actor_rng,
            inputs=(
                jnp.ones((1, act_dim)),
                jnp.zeros((1, 1)),
                jnp.ones((1, obs_dim)),
            ),
            x_dim=act_dim,
            steps=cfg.flow.steps,
            optimizer=optax.adam(learning_rate=cfg.flow.lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

        if cfg.critic_backbone_cls == "mlp":
            critic_backbone = MLP(
                hidden_dims=cfg.critic_hidden_dims,
                activation=get_activation(cfg.critic_activation),
                layer_norm=cfg.critic_layer_norm,
                kernel_init=critic_initializer,
            )
        elif cfg.critic_backbone_cls == "flac":
            critic_backbone = FLACCriticBackbone(
                hidden_dims=cfg.critic_hidden_dims,
                activation=get_activation(cfg.critic_activation),
                layer_norm=cfg.critic_layer_norm,
                kernel_init=critic_initializer,
            )
        else:
            raise ValueError(
                f"Unsupported FLAC critic backbone: {cfg.critic_backbone_cls}"
            )

        if cfg.distributional_critic:
            critic_base = CategoricalCritic(
                backbone=critic_backbone,
                output_nodes=cfg.critic_num_atoms,
                kernel_init=critic_initializer,
            )
        else:
            critic_base = ScalarCritic(
                backbone=critic_backbone,
                kernel_init=critic_initializer,
            )
        critic_def = Ensemblize(
            base=critic_base,
            ensemble_size=cfg.critic_ensemble_size,
        )
        critic_inputs = (
            jnp.ones((1, obs_dim)),
            jnp.ones((1, act_dim)),
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=critic_inputs,
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=critic_inputs,
        )
        self.log_alpha = Model.create(
            TunableCoefficient(init_value=cfg.init_log_alpha),
            alpha_rng,
            inputs=(),
            optimizer=optax.adam(learning_rate=cfg.alpha_lr),
        )

        self.atoms = jnp.linspace(
            cfg.critic_v_min,
            cfg.critic_v_max,
            cfg.critic_num_atoms,
        )
        self.target_kinetic = cfg.target_kinetic_coef * act_dim
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        if self.cfg.distributional_critic:
            self.rng, self.critic, metrics = update_categorical_critic(
                self.rng,
                self.actor,
                self.critic,
                self.critic_target,
                self.log_alpha,
                batch,
                self.atoms,
                discount=self.cfg.discount,
                reward_scale=self.cfg.reward_scale,
                v_min=self.cfg.critic_v_min,
                v_max=self.cfg.critic_v_max,
            )
        else:
            self.rng, self.critic, metrics = update_scalar_critic(
                self.rng,
                self.actor,
                self.critic,
                self.critic_target,
                self.log_alpha,
                batch,
                discount=self.cfg.discount,
                reward_scale=self.cfg.reward_scale,
            )

        if self._n_training_steps % self.cfg.update_actor_every == 0:
            self.rng, self.actor, kinetic, actor_metrics = update_actor(
                self.rng,
                self.actor,
                self.critic,
                self.log_alpha,
                batch,
                self.atoms,
                distributional_critic=self.cfg.distributional_critic,
            )
            self.log_alpha, alpha_metrics = update_alpha(
                self.log_alpha,
                kinetic,
                self.target_kinetic,
            )
            self.critic_target = ema_update(
                self.critic,
                self.critic_target,
                self.cfg.ema,
            )
            metrics.update(actor_metrics)
            metrics.update(alpha_metrics)

        self._n_training_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        assert num_samples == 1, "FLAC only supports num_samples=1"
        self.rng, action, _ = jit_sample_actions(
            self.rng,
            self.actor,
            obs,
        )
        return action, {}
