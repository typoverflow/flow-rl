from functools import partial
from typing import Callable, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.dime import DIMEConfig
from flowrl.flow.dime import DIMEFlow, DIMEScoreNetwork
from flowrl.functional.activation import get_activation
from flowrl.functional.categorical import project_categorical
from flowrl.functional.ema import ema_update
from flowrl.module.critic import CategoricalCritic
from flowrl.module.misc import PositiveTunableCoefficient
from flowrl.module.model import BatchStatModel, Model
from flowrl.module.normalization import BatchRenorm
from flowrl.types import Batch, Metric, Param, PRNGKey, Sequence


class DIMECriticBackbone(nn.Module):
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray]
    use_batch_renorm: bool
    batch_norm_momentum: float
    batch_norm_epsilon: float
    batch_norm_warmup: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        def normalize(value: jnp.ndarray, name: str) -> jnp.ndarray:
            if self.use_batch_renorm:
                return BatchRenorm(
                    momentum=self.batch_norm_momentum,
                    epsilon=self.batch_norm_epsilon,
                    warmup_steps=self.batch_norm_warmup,
                    name=name,
                )(value, training=training)
            dummy = self.variable(
                "batch_stats",
                f"{name}_dummy",
                lambda: jnp.array(0.0),
            )
            return value + 0.0 * dummy.value

        x = normalize(x, "input_norm")
        for index, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)
            x = normalize(x, f"hidden_norm_{index}")
        return x


class DIMECriticEnsemble(nn.Module):
    base: nn.Module
    ensemble_size: int

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        def call_base(base_instance, obs, action, training):
            return base_instance(obs, action, training=training)

        vmap_fn = nn.vmap(
            call_base,
            variable_axes={"params": 0, "batch_stats": 0},
            split_rngs={
                "params": True,
                "dropout": True,
                "batch_stats": True,
            },
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        logits = vmap_fn(self.base, obs, action, training)
        return jax.nn.softmax(logits, axis=-1)


@partial(
    jax.jit,
    static_argnames=(
        "discount",
        "v_min",
        "v_max",
        "critic_entropy_coef",
    ),
)
def update_critic(
    rng: PRNGKey,
    actor_target: DIMEFlow,
    critic: BatchStatModel,
    temperature: Model,
    batch: Batch,
    atoms: jnp.ndarray,
    discount: float,
    v_min: float,
    v_max: float,
    critic_entropy_coef: float,
) -> Tuple[PRNGKey, BatchStatModel, Metric]:
    rng, action_rng = jax.random.split(rng)
    (
        _,
        next_action,
        next_running_cost,
        next_stochastic_cost,
        next_terminal_cost,
        _,
    ) = actor_target.sample(action_rng, batch.next_obs)
    next_action = jax.lax.stop_gradient(next_action)
    total_next_cost = jax.lax.stop_gradient(
        next_running_cost + next_stochastic_cost + next_terminal_cost
    )
    alpha = temperature()
    bootstrap = discount * (1 - batch.terminal)
    entropy_bonus = -bootstrap * alpha * total_next_cost

    def critic_loss_fn(
        critic_params: Param,
        batch_stats,
        dropout_rng: PRNGKey,
    ):
        (all_probabilities, updates) = critic.apply(
            {
                "params": critic_params,
                "batch_stats": batch_stats,
            },
            jnp.concatenate([batch.obs, batch.next_obs], axis=0),
            jnp.concatenate([batch.action, next_action], axis=0),
            training=True,
            mutable=["batch_stats"],
            rngs={"dropout": dropout_rng},
        )
        current_probabilities, next_probabilities = jnp.split(
            all_probabilities,
            2,
            axis=1,
        )
        target_atoms = (
            batch.reward
            + entropy_bonus
            + bootstrap * atoms[jnp.newaxis, :]
        )
        projected_targets = jax.vmap(
            project_categorical,
            in_axes=(0, None, None, None),
        )(
            next_probabilities,
            target_atoms,
            v_min,
            v_max,
        )
        target_distribution = jax.lax.stop_gradient(
            projected_targets.mean(axis=0)
        )

        log_probabilities = jnp.log(current_probabilities + 1e-15)
        cross_entropy = -jnp.sum(
            target_distribution[jnp.newaxis, ...] * log_probabilities,
            axis=-1,
        ).mean(axis=-1)
        distribution_regularizer = jnp.sum(
            current_probabilities * log_probabilities,
            axis=-1,
        ).mean(axis=-1)
        per_critic_loss = (
            cross_entropy
            + critic_entropy_coef * distribution_regularizer
        )
        critic_loss = per_critic_loss.sum()

        current_q = jnp.sum(
            current_probabilities * atoms,
            axis=-1,
            keepdims=True,
        )
        clipped_fraction = jnp.mean(
            (target_atoms < v_min) | (target_atoms > v_max)
        )
        metrics = {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": current_q.mean(),
            "misc/target_q_mean": (
                target_distribution * atoms
            ).sum(axis=-1).mean(),
            "misc/reward": batch.reward.mean(),
            "misc/temperature": alpha,
            "misc/next_running_cost": next_running_cost.mean(),
            "misc/next_terminal_cost": next_terminal_cost.mean(),
            "misc/c51_mass": target_distribution.sum(axis=-1).mean(),
            "misc/c51_clipped_fraction": clipped_fraction,
            "misc/critic_entropy": (
                -current_probabilities * log_probabilities
            ).sum(axis=-1).mean(),
        }
        return critic_loss, (metrics, updates["batch_stats"])

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)
    return rng, new_critic, metrics


@jax.jit
def update_actor(
    rng: PRNGKey,
    actor: DIMEFlow,
    critic: BatchStatModel,
    temperature: Model,
    batch: Batch,
    atoms: jnp.ndarray,
) -> Tuple[PRNGKey, DIMEFlow, jnp.ndarray, Metric]:
    rng, actor_rng = jax.random.split(rng)

    def actor_loss_fn(
        actor_params: Param,
        dropout_rng: PRNGKey,
    ):
        sample_rng = jax.random.fold_in(actor_rng, dropout_rng[0])
        (
            _,
            action,
            running_cost,
            stochastic_cost,
            terminal_cost,
            latent_path,
        ) = actor.sample(
            sample_rng,
            batch.obs,
            training=True,
            params=actor_params,
        )
        probabilities = critic(
            batch.obs,
            action,
            training=False,
        )
        q = jnp.sum(
            probabilities * atoms,
            axis=-1,
            keepdims=True,
        ).mean(axis=0)
        total_cost = running_cost + stochastic_cost + terminal_cost
        actor_loss = (
            -q + temperature() * total_cost
        ).mean()
        metrics = {
            "loss/actor_loss": actor_loss,
            "misc/actor_q": q.mean(),
            "misc/running_cost": running_cost.mean(),
            "misc/stochastic_cost": stochastic_cost.mean(),
            "misc/terminal_cost": terminal_cost.mean(),
            "misc/latent_max": latent_path.max(),
            "misc/latent_min": latent_path.min(),
            "misc/latent_mean": latent_path.mean(),
            "misc/action_l1": jnp.abs(action).mean(),
        }
        return actor_loss, metrics

    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    return rng, new_actor, metrics["misc/running_cost"], metrics


@jax.jit
def update_temperature(
    temperature: Model,
    running_cost: jnp.ndarray,
    target_entropy: float,
) -> Tuple[Model, Metric]:
    running_cost = jax.lax.stop_gradient(running_cost)

    def temperature_loss_fn(
        temperature_params: Param,
        dropout_rng: PRNGKey,
    ):
        alpha = temperature.apply({"params": temperature_params})
        loss = -alpha * (running_cost - target_entropy)
        return loss, {
            "loss/temperature_loss": loss,
            "misc/temperature": alpha,
        }

    return temperature.apply_gradient(temperature_loss_fn)


class DIMEAgent(BaseAgent):
    """Diffusion-Based Maximum Entropy Reinforcement Learning.

    The official CrossQ path bootstraps from the live critic, so DIME does not
    keep a redundant target critic.
    """

    name = "DIMEAgent"
    model_names = ["actor", "actor_target", "critic", "temperature"]

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: DIMEConfig,
        seed: int,
    ):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        if cfg.critic_ensemble_size != 2:
            raise ValueError("DIME requires exactly two critics")
        if cfg.critic_num_atoms < 2:
            raise ValueError("DIME requires at least two critic atoms")
        if cfg.critic_v_min >= cfg.critic_v_max:
            raise ValueError("DIME critic support is invalid")
        self.rng, actor_rng, critic_rng, temperature_rng = jax.random.split(
            self.rng,
            4,
        )

        score_network = DIMEScoreNetwork(
            action_dim=act_dim,
            hidden_dims=cfg.sampler.hidden_dims,
            activation=get_activation(cfg.sampler.activation),
            time_dim=cfg.sampler.time_dim,
            layer_norm=cfg.sampler.layer_norm,
            initial_dt=cfg.sampler.dt,
            initial_friction=cfg.sampler.friction,
            per_dim_friction=cfg.sampler.per_dim_friction,
            output_clip=cfg.sampler.output_clip,
            output_kernel_init=cfg.sampler.output_kernel_init,
        )
        actor_optimizer = optax.chain(
            optax.zero_nans(),
            optax.clip(cfg.actor_grad_clip),
            optax.adam(
                cfg.actor_lr,
                b1=cfg.adam_b1,
                b2=cfg.adam_b2,
            ),
        )
        actor_inputs = (
            jnp.ones((1, act_dim)),
            jnp.ones((1, obs_dim)),
            jnp.zeros((1, 1)),
        )
        actor_kwargs = dict(
            network=score_network,
            inputs=actor_inputs,
            x_dim=act_dim,
            steps=cfg.sampler.steps,
            prior_std=cfg.sampler.prior_std,
            learn_dt=cfg.sampler.learn_dt,
            learn_friction=cfg.sampler.learn_friction,
            schedule_min=cfg.sampler.schedule_min,
            schedule_offset=cfg.sampler.schedule_offset,
            schedule_power=cfg.sampler.schedule_power,
        )
        self.actor = DIMEFlow.create(
            rng=actor_rng,
            optimizer=actor_optimizer,
            **actor_kwargs,
        )
        self.actor_target = DIMEFlow.create(
            rng=actor_rng,
            **actor_kwargs,
        )

        critic_backbone = DIMECriticBackbone(
            hidden_dims=cfg.critic_hidden_dims,
            activation=get_activation(cfg.critic_activation),
            use_batch_renorm=cfg.batch_renorm,
            batch_norm_momentum=cfg.batch_norm_momentum,
            batch_norm_epsilon=cfg.batch_norm_epsilon,
            batch_norm_warmup=cfg.batch_norm_warmup,
        )
        critic_network = DIMECriticEnsemble(
            base=CategoricalCritic(
                backbone=critic_backbone,
                output_nodes=cfg.critic_num_atoms,
                kernel_init=nn.initializers.lecun_normal,
            ),
            ensemble_size=cfg.critic_ensemble_size,
        )
        self.critic = BatchStatModel.create(
            network=critic_network,
            rng=critic_rng,
            inputs=(
                jnp.ones((1, obs_dim)),
                jnp.ones((1, act_dim)),
            ),
            input_kwargs={"training": False},
            optimizer=optax.adam(
                cfg.critic_lr,
                b1=cfg.adam_b1,
                b2=cfg.adam_b2,
            ),
        )
        self.temperature = Model.create(
            PositiveTunableCoefficient(
                init_value=cfg.init_temperature,
            ),
            temperature_rng,
            inputs=(),
            optimizer=optax.adam(cfg.temperature_lr),
        )
        self.atoms = jnp.linspace(
            cfg.critic_v_min,
            cfg.critic_v_max,
            cfg.critic_num_atoms,
        )
        self.target_entropy = cfg.target_entropy_scale * act_dim
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.critic, metrics = update_critic(
            self.rng,
            self.actor_target,
            self.critic,
            self.temperature,
            batch,
            self.atoms,
            discount=self.cfg.discount,
            v_min=self.cfg.critic_v_min,
            v_max=self.cfg.critic_v_max,
            critic_entropy_coef=self.cfg.critic_entropy_coef,
        )
        self._n_training_steps += 1
        if self._n_training_steps % self.cfg.actor_update_freq == 0:
            (
                self.rng,
                self.actor,
                running_cost,
                actor_metrics,
            ) = update_actor(
                self.rng,
                self.actor,
                self.critic,
                self.temperature,
                batch,
                self.atoms,
            )
            self.temperature, temperature_metrics = update_temperature(
                self.temperature,
                running_cost,
                self.target_entropy,
            )
            self.actor_target = ema_update(
                self.actor,
                self.actor_target,
                self.cfg.target_actor_ema,
            )
            metrics.update(actor_metrics)
            metrics.update(temperature_metrics)
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        assert num_samples == 1, "DIME only supports num_samples=1"
        self.rng, sample_rng = jax.random.split(self.rng)
        _, action, _, _, _, _ = self.actor.sample(sample_rng, obs)
        return action, {}
