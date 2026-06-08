from functools import partial

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.ppo import (
    adaptive_lr_update,
    apply_lr_to_model,
    clipped_value_loss,
    compute_gae,
    make_minibatch_indices,
    normalize_advantages,
    ppo_surrogate_loss,
)
from flowrl.config.online.algo.genpo import GenPOConfig
from flowrl.flow.cnf import FlowBackbone
from flowrl.flow.genpo import GenPOFlow
from flowrl.functional.activation import get_activation
from flowrl.module.critic import ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Metric, PRNGKey, RolloutBatch

# ======= Sampling ========

@partial(jax.jit, static_argnames=("deterministic",))
def jit_sample_action_genpo(rng, actor, obs, deterministic):
    rng, x0_rng = jax.random.split(rng)
    B = obs.shape[0]
    aug_dim = 2 * actor.a_dim

    if deterministic:
        x0 = jnp.zeros((B, aug_dim))
        x1 = actor.forward(obs, x0)
        log_prob = jnp.zeros(B)
    else:
        x0 = jax.random.normal(x0_rng, (B, aug_dim))
        x1 = actor.forward(obs, x0)
        log_prob = actor.log_prob(obs, x0)

    # Env action is the average of the two augmented halves (official impl).
    action = 0.5 * x1[:, :actor.a_dim] + 0.5 * x1[:, actor.a_dim:]

    return action, log_prob, x1


# ======= Training Update ========

@partial(jax.jit, static_argnames=(
    "gamma", "gae_lambda", "clip_epsilon", "entropy_coeff", "compress_coef",
    "value_loss_coef", "use_clipped_value_loss", "reward_scaling", "normalize_advantage",
    "adaptive", "desired_kl", "num_epochs", "num_minibatches", "batch_size",
))
def jit_update_genpo(
    rng, actor, critic, rollout, actor_lr, critic_lr,
    gamma, gae_lambda, clip_epsilon, entropy_coeff, compress_coef,
    value_loss_coef, use_clipped_value_loss, reward_scaling, normalize_advantage,
    adaptive, desired_kl, num_epochs, num_minibatches, batch_size,
):
    T, B = rollout.rewards.shape[:2]

    # Values computed with rollout-time critic (no updates yet this step).
    old_values = critic(rollout.obs)            # (T, B, 1)
    next_old_values = critic(rollout.next_obs)  # (T, B, 1)

    lambda_returns, advantages = jax.lax.stop_gradient(
        compute_gae(
            terminated=rollout.terminated, truncated=rollout.truncated,
            rewards=rollout.rewards * reward_scaling,
            values=old_values, next_values=next_old_values,
            gae_lambda=gae_lambda, gamma=gamma,
        )
    )

    if normalize_advantage:
        advantages = normalize_advantages(advantages)

    flat_obs = rollout.obs.reshape(T * B, -1)
    flat_actions = rollout.actions.reshape(T * B, -1)
    flat_advantages = advantages.reshape(T * B, 1)
    flat_returns = lambda_returns.reshape(T * B, 1)
    flat_old_values = old_values.reshape(T * B, 1)
    flat_old_log_probs = rollout.extras["log_prob"].reshape(T * B, 1)
    flat_aug_actions = rollout.extras["aug_action"].reshape(T * B, -1)

    def epoch_step(carry, _):
        rng, actor, critic, actor_lr, critic_lr = carry
        rng, perm_rng = jax.random.split(rng)
        mb_indices = make_minibatch_indices(perm_rng, T * B, num_minibatches, batch_size)

        def minibatch_step(carry, indices):
            rng, actor, critic, actor_lr, critic_lr = carry
            rng, compress_rng = jax.random.split(rng)

            mb_obs = flat_obs[indices]
            mb_advantages = flat_advantages[indices]
            mb_returns = flat_returns[indices]
            mb_old_values = flat_old_values[indices]
            mb_old_log_probs = flat_old_log_probs[indices]
            mb_aug_actions = flat_aug_actions[indices]

            compress_x0 = jax.random.normal(compress_rng, (batch_size, 2 * actor.a_dim))

            # --- Actor loss -----------------------------------------------------
            def actor_loss_fn(actor_params, dropout_rng):
                # Log prob via inverse Jacobian (matches official impl).
                new_log_prob = actor.log_prob_via_inverse(
                    mb_obs, mb_aug_actions, params=actor_params,
                )[:, jnp.newaxis]  # (batch_size, 1)

                policy_loss, ratio, clipped_frac = ppo_surrogate_loss(
                    new_log_prob, mb_old_log_probs, mb_advantages, clip_epsilon
                )

                # Entropy loss (maximize entropy = minimize mean log_prob).
                if entropy_coeff > 0:
                    entropy_loss = entropy_coeff * jnp.mean(new_log_prob)
                else:
                    entropy_loss = 0.0

                # Compress loss on fresh forward samples (L2 norm, matches official).
                if compress_coef > 0:
                    x1_fresh = actor.forward(mb_obs, compress_x0, params=actor_params)
                    z_f = x1_fresh[:, :actor.a_dim]
                    y_f = x1_fresh[:, actor.a_dim:]
                    compress_loss = compress_coef * jnp.mean(
                        jnp.sqrt(jnp.sum((z_f - y_f) ** 2, axis=-1) + 1e-8)
                    )
                else:
                    compress_loss = 0.0

                total_loss = policy_loss + entropy_loss + compress_loss

                # KL estimate via importance ratio (official adaptive schedule).
                kl_mean = jnp.mean(jax.lax.stop_gradient(mb_old_log_probs - new_log_prob))

                return total_loss, {
                    "loss/policy_loss": policy_loss,
                    "loss/entropy_loss": entropy_loss,
                    "loss/compress_loss": compress_loss,
                    "misc/entropy": -jnp.mean(new_log_prob),
                    "misc/policy_ratio": ratio.mean(),
                    "misc/clipped_ratio": clipped_frac,
                    "misc/log_prob_mean": jnp.mean(new_log_prob),
                    "misc/kl_mean": kl_mean,
                }

            new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

            # --- Critic loss ----------------------------------------------------
            def critic_loss_fn(critic_params, dropout_rng):
                v = critic.apply(
                    {"params": critic_params}, mb_obs,
                    training=True, rngs={"dropout": dropout_rng},
                )
                v_loss_raw = clipped_value_loss(
                    v, mb_old_values, mb_returns, clip_epsilon, use_clipped_value_loss
                )
                return value_loss_coef * v_loss_raw, {
                    "loss/value_loss": v_loss_raw,
                    "misc/value_mean": jnp.mean(v),
                }

            new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

            # --- Adaptive LR schedule (1-step lag) ------------------------------
            # KL is read from the actor aux (new_log_prob already computed for the
            # gradient); the updated LR applies to the *next* minibatch, avoiding a
            # second expensive jacrev probe.
            if adaptive:
                kl_mean = actor_metrics["misc/kl_mean"]
                actor_lr, critic_lr = adaptive_lr_update(
                    kl_mean, actor_lr, critic_lr, desired_kl
                )
                new_actor = apply_lr_to_model(new_actor, actor_lr)
                new_critic = apply_lr_to_model(new_critic, critic_lr)

            metrics = {
                **actor_metrics,
                **critic_metrics,
                "misc/actor_lr": actor_lr,
                "misc/critic_lr": critic_lr,
            }
            return (rng, new_actor, new_critic, actor_lr, critic_lr), metrics

        (rng, actor, critic, actor_lr, critic_lr), mb_metrics = jax.lax.scan(
            minibatch_step,
            init=(rng, actor, critic, actor_lr, critic_lr),
            xs=mb_indices,
        )
        return (rng, actor, critic, actor_lr, critic_lr), mb_metrics

    (rng, actor, critic, actor_lr, critic_lr), all_metrics = jax.lax.scan(
        epoch_step,
        init=(rng, actor, critic, actor_lr, critic_lr),
        length=num_epochs,
    )

    metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
    metrics.update({
        "misc/reward_mean": rollout.rewards.mean(),
        "misc/obs_mean": flat_obs.mean(),
        "misc/obs_std": flat_obs.std(axis=0).mean(),
        "misc/action_l1_mean": jnp.abs(flat_actions).mean(),
        "misc/advantages_mean": flat_advantages.mean(),
        "misc/advantages_std": flat_advantages.std(axis=0).mean(),
        "misc/returns_mean": flat_returns.mean(),
    })

    return rng, actor, critic, actor_lr, critic_lr, metrics


# ======= Agent ========

class GenPOAgent(BaseAgent):
    """
    Generative Policy Optimization (GenPO)
    Uses coupled Heun solver on augmented action space with exact log-probability
    via Jacobian determinant, combined with PPO clipping.
    """
    name = "GenPOAgent"
    model_names = ["actor", "critic"]

    def __init__(self, obs_dim, act_dim, cfg: GenPOConfig, seed):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        actor_activation = get_activation(cfg.flow.activation)
        critic_activation = get_activation(cfg.critic_activation)
        backbone_cls = {"mlp": MLP, "simba": Simba}[cfg.backbone_cls]

        flow_backbone = FlowBackbone(
            vel_predictor=backbone_cls(
                hidden_dims=cfg.flow.hidden_dims,
                activation=actor_activation,
                output_dim=act_dim,
            ),
            time_embedding=LearnableFourierEmbedding(output_dim=cfg.flow.time_dim),
        )
        # Use inject_hyperparams so the adaptive schedule can mutate LR inside jit.
        actor_opt = optax.inject_hyperparams(optax.adam)(learning_rate=cfg.flow.lr)
        self.actor = GenPOFlow.create(
            network=flow_backbone,
            rng=actor_rng,
            inputs=(
                jnp.ones((1, act_dim)),
                jnp.ones((1, 1)),
                jnp.ones((1, obs_dim)),
            ),
            a_dim=act_dim,
            steps=cfg.flow.steps,
            mix_para=cfg.flow.mix_para,
            optimizer=actor_opt,
            clip_grad_norm=cfg.clip_grad_norm,
        )

        critic_def = ScalarCritic(
            backbone=backbone_cls(
                hidden_dims=cfg.critic_hidden_dims,
                activation=critic_activation,
            ),
        )
        critic_opt = optax.inject_hyperparams(optax.adam)(learning_rate=cfg.critic_lr)
        self.critic = Model.create(
            critic_def, critic_rng,
            inputs=(jnp.ones((1, obs_dim)),),
            optimizer=critic_opt,
            clip_grad_norm=cfg.clip_grad_norm,
        )

        # Running LRs; mutated in-place by the adaptive schedule each train_step.
        self.actor_lr = jnp.asarray(cfg.flow.lr, dtype=jnp.float32)
        self.critic_lr = jnp.asarray(cfg.critic_lr, dtype=jnp.float32)

    def train_step(self, rollout: RolloutBatch, step: int) -> Metric:
        (
            self.rng,
            self.actor,
            self.critic,
            self.actor_lr,
            self.critic_lr,
            metrics,
        ) = jit_update_genpo(
            self.rng, self.actor, self.critic, rollout,
            self.actor_lr, self.critic_lr,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_epsilon=self.cfg.clip_epsilon,
            entropy_coeff=self.cfg.entropy_coeff,
            compress_coef=self.cfg.compress_coef,
            value_loss_coef=self.cfg.value_loss_coef,
            use_clipped_value_loss=self.cfg.use_clipped_value_loss,
            reward_scaling=self.cfg.reward_scaling,
            normalize_advantage=self.cfg.normalize_advantage,
            adaptive=(self.cfg.schedule == "adaptive"),
            desired_kl=self.cfg.desired_kl,
            num_epochs=self.cfg.num_epochs,
            num_minibatches=self.cfg.num_minibatches,
            batch_size=self.cfg.batch_size,
        )
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        assert num_samples == 1
        self.rng, sample_rng = jax.random.split(self.rng)

        action, log_prob, aug_action = jit_sample_action_genpo(
            sample_rng, self.actor, obs, deterministic,
        )

        return action, {
            "log_prob": log_prob[:, jnp.newaxis],
            "aug_action": aug_action,
        }
