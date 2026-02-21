from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.dacer import DACERConfig
from flowrl.flow.continuous_ddpm import ContinuousDDPM, ContinuousDDPMBackbone
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import Ensemblize, GaussianCritic
from flowrl.module.misc import TunableCoefficient
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("deterministic", "num_samples", "noise_scaler"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    log_alpha: Model,
    obs: jnp.ndarray,
    deterministic: bool,
    num_samples: int,
    noise_scaler: float,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, xT_rng, noise_rng = jax.random.split(rng, 3)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], actor.x_dim))
    rng, actions, _ = actor.sample(rng, xT, obs_repeat)
    if num_samples == 1:
        actions = actions[:, 0]
    if not deterministic:
        actions = actions + \
            jax.random.normal(noise_rng, actions.shape) * jnp.exp(log_alpha()) * noise_scaler
    return rng, actions.clip(-1.0, 1.0)


@partial(jax.jit, static_argnames=("discount", "ema", "reward_scale", "update_actor", "noise_scaler"))
def jit_update_dacer(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    critic: Model,
    critic_target: Model,
    log_alpha: Model,
    running_mean_q_std: jnp.ndarray,
    batch: Batch,
    discount: float,
    ema: float,
    reward_scale: float,
    update_actor: bool,
    noise_scaler: float,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    batch.reward = batch.reward * reward_scale

    def q_evaluate(rng: PRNGKey, model: Model, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        rng, evaluate_rng = jax.random.split(rng)
        mean, std = model(obs, action)
        z = jax.random.normal(evaluate_rng, mean.shape)
        z = jnp.clip(z, -3.0, 3.0)
        q_value = mean + std * z
        return rng, mean, std, q_value


    rng, next_action = jit_sample_actions(
        rng,
        actor,
        log_alpha,
        batch.next_obs,
        deterministic=False,
        num_samples=1,
        noise_scaler=noise_scaler,
    )
    rng, q_mean_target, q_std_target, q_target = q_evaluate(
        rng,
        critic_target,
        batch.next_obs,
        next_action,
    )
    q_mean_target = q_mean_target.min(axis=0)
    q_target = jnp.where(q_mean_target[0] < q_mean_target[1], q_target[0], q_target[1])
    q_mean_target = batch.reward + discount * (1 - batch.terminal) * q_mean_target
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        both_q_mean, both_q_std = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            rngs={"dropout": dropout_rng},
        )

        def huber_loss(pred: jnp.ndarray, target: jnp.ndarray, delta: float) -> jnp.ndarray:
            error = pred - target
            return jnp.where(
                jnp.abs(error) <= delta,
                0.5 * jnp.square(error),
                delta * (jnp.abs(error) - 0.5 * delta)
            )

        def loss_fn(q_mean: jnp.ndarray, q_std: jnp.ndarray) -> jnp.ndarray:
            q_target_bounded = jax.lax.stop_gradient(
                q_mean + jnp.clip(q_target - q_mean, -3.0 * running_mean_q_std, 3.0 * running_mean_q_std)
            )
            q_std_detach = jax.lax.stop_gradient(jnp.maximum(q_std, 1e-6))
            bias = 0.1
            ratio = jnp.square(running_mean_q_std) / (jnp.square(q_std_detach) + bias)
            ratio = jax.lax.clamp(0.1, ratio, 10.0)
            huber_term = huber_loss(q_mean, q_mean_target, delta=50)
            std_huber_term = huber_loss(jax.lax.stop_gradient(q_mean), q_target_bounded, delta=50)
            std_term = q_std * (jnp.square(q_std_detach) - std_huber_term) / (q_std_detach + bias)
            q_loss = jnp.mean(ratio * (huber_term + std_term))
            return q_loss

        both_q_loss = jax.vmap(
            loss_fn,
            in_axes=(0, 0),
            out_axes=0,
        )(both_q_mean, both_q_std)

        return both_q_loss.sum(axis=0), {
            "loss/critic_loss": both_q_loss.sum(axis=0),
            "misc/q_std": both_q_std.mean(),
            "misc/q_mean": both_q_mean.mean(),
        }

    new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    new_running_mean_q_std = ema * critic_metrics["misc/q_std"] + (1-ema) * running_mean_q_std

    # update actor
    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        dropout_rng, rng1, rng2 = jax.random.split(dropout_rng, 3)
        new_xT = jax.random.normal(rng1, batch.action.shape)
        rng2, new_action, _ = actor.sample(
            rng2,
            new_xT,
            batch.obs,
            training=True,
            params=actor_params,
        )

        rng2, new_q_mean, new_q_std, new_q_value = q_evaluate(
            rng2,
            critic,
            batch.obs,
            new_action,
        )
        actor_loss = - new_q_mean.min(axis=0).mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss,
            "misc/next_action_l1": jnp.abs(new_action).mean(),
        }

    if update_actor:
        new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)
    else:
        new_actor, actor_metrics = actor, {}

    return rng, new_actor, new_critic, new_critic_target, new_running_mean_q_std, {
        **critic_metrics,
        **actor_metrics,
        "misc/running_mean_q_std": new_running_mean_q_std,
    }


@partial(jax.jit, static_argnames=("entropy_num_samples", "target_entropy", "noise_scaler"))
def jit_update_alpha(
    rng: PRNGKey,
    actor: ContinuousDDPM,
    log_alpha: Model,
    batch: Batch,
    entropy_num_samples: int,
    target_entropy: float,
    noise_scaler: float,
) -> Tuple[PRNGKey, Model, Metric]:
    rng, action_samples = jit_sample_actions(
        rng,
        actor,
        log_alpha,
        batch.obs,
        deterministic=False,
        num_samples=entropy_num_samples,
        noise_scaler=noise_scaler,
    )

    def estimate_entropy(actions: jnp.ndarray) -> float:
        # use gaussian mixture model to estimate the entropy
        import numpy as np
        from sklearn.mixture import GaussianMixture
        total_entropy = []
        for action in actions:
            gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
            gmm.fit(action)
            weights = gmm.weights_
            entropies = []
            for i in range(gmm.n_components):
                cov_matrix = gmm.covariances_[i]
                d = cov_matrix.shape[0]
                entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov_matrix)[1]
                entropies.append(entropy)
            entropy = -np.sum(weights * np.log(weights)) + np.sum(weights * np.array(entropies))
            total_entropy.append(entropy)
        final_entropy = sum(total_entropy) / len(total_entropy)
        return final_entropy

    def alpha_loss_fn(log_alpha_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        log_alpha_value = log_alpha.apply(
            {"params": log_alpha_params},
        )
        loss = log_alpha_value * (entropy - target_entropy).mean()
        return loss, {
            "loss/alpha_loss": loss,
            "misc/entropy": entropy,
            "misc/alpha": jnp.exp(log_alpha_value),
        }

    entropy = jax.pure_callback(estimate_entropy, jax.ShapeDtypeStruct((), jnp.float32), action_samples)
    new_log_alpha, metrics = log_alpha.apply_gradient(alpha_loss_fn)

    return rng, new_log_alpha, metrics


class DACERAgent(BaseAgent):
    """
    Diffusion Actor Critic with Entropy Regulator (DACER)
    https://arxiv.org/abs/2405.15177
    """
    name = "DACERAgent"
    model_names = ["actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DACERConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng, log_alpha_rng = jax.random.split(self.rng, 4)

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
            base_cls=GaussianCritic,
            base_kwargs=dict(
                backbone=MLP(
                    hidden_dims=cfg.critic_hidden_dims,
                    activation=critic_activation,
                    layer_norm=False,
                    dropout=None,
                ),
            ),
            ensemble_size=cfg.critic_ensemble_size,
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
        self.log_alpha = Model.create(
            TunableCoefficient(init_value=jnp.log(3.0)),
            rng=log_alpha_rng,
            inputs=(),
            optimizer=optax.adam(learning_rate=cfg.alpha_lr),
        )
        self.running_mean_q_std = jnp.array(1.0)
        self.target_entropy = -0.9 * self.act_dim

        # define tracking variables
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.critic, self.critic_target, self.running_mean_q_std, metrics = jit_update_dacer(
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            self.log_alpha,
            self.running_mean_q_std,
            batch,
            self.cfg.discount,
            self.cfg.ema,
            self.cfg.reward_scale,
            self._n_training_steps % self.cfg.update_actor_every == 0,
            self.cfg.noise_scaler,
        )
        if self._n_training_steps % self.cfg.update_alpha_every == 0:
            self.rng, self.log_alpha, alpha_metrics = jit_update_alpha(
                self.rng,
                self.actor,
                self.log_alpha,
                batch,
                self.cfg.entropy_num_samples,
                self.target_entropy,
                self.cfg.noise_scaler,
            )
            metrics.update(alpha_metrics)
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
            self.actor,
            self.log_alpha,
            obs,
            deterministic=deterministic,
            num_samples=1,
            noise_scaler=self.cfg.noise_scaler,
        )
        return action, {}
