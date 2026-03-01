from functools import partial

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.ppo import compute_gae
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
        action = x1[:, :actor.a_dim]
        log_prob = jnp.zeros(B)
    else:
        x0 = jax.random.normal(x0_rng, (B, aug_dim))
        x1 = actor.forward(obs, x0)
        log_prob = actor.log_prob(obs, x0)
        action = x1[:, :actor.a_dim]

    return action, log_prob, x1


# ======= Training Update ========

@partial(jax.jit, static_argnames=(
    "gamma", "gae_lambda", "clip_epsilon", "entropy_coeff", "compress_coef",
    "reward_scaling", "normalize_advantage",
    "num_epochs", "num_minibatches", "batch_size",
))
def jit_update_genpo(
    rng, actor, critic, rollout,
    gamma, gae_lambda, clip_epsilon, entropy_coeff, compress_coef,
    reward_scaling, normalize_advantage,
    num_epochs, num_minibatches, batch_size,
):
    T, B = rollout.rewards.shape[:2]

    value_pred = critic(rollout.obs)
    next_value_pred = critic(rollout.next_obs)

    gae_vs, gae_advantages = jax.lax.stop_gradient(
        compute_gae(
            terminated=rollout.terminated, truncated=rollout.truncated,
            rewards=rollout.rewards * reward_scaling,
            values=value_pred, next_values=next_value_pred,
            gae_lambda=gae_lambda, gamma=gamma,
        )
    )

    if normalize_advantage:
        gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)

    flat_obs = rollout.obs.reshape(T * B, -1)
    flat_actions = rollout.actions.reshape(T * B, -1)
    flat_advantages = gae_advantages.reshape(T * B, 1)
    flat_gae_vs = gae_vs.reshape(T * B, 1)
    flat_truncations = rollout.truncated.reshape(T * B, 1)
    flat_old_log_probs = rollout.extras["log_prob"].reshape(T * B, 1)
    flat_aug_actions = rollout.extras["aug_action"].reshape(T * B, -1)

    def epoch_step(carry, _):
        rng, actor, critic = carry
        rng, perm_rng = jax.random.split(rng)

        perm = jax.random.permutation(perm_rng, T * B)
        total = num_minibatches * batch_size
        perm = perm[:total]
        mb_indices = perm.reshape(num_minibatches, batch_size)

        def minibatch_step(carry, indices):
            rng, actor, critic = carry
            rng, compress_rng = jax.random.split(rng)

            mb_obs = flat_obs[indices]
            mb_advantages = flat_advantages[indices]
            mb_gae_vs = flat_gae_vs[indices]
            mb_truncations = flat_truncations[indices]
            mb_old_log_probs = flat_old_log_probs[indices]
            mb_aug_actions = flat_aug_actions[indices]

            compress_x0 = jax.random.normal(compress_rng, (batch_size, 2 * actor.a_dim))

            def actor_loss_fn(actor_params, dropout_rng):
                # Log prob via inverse Jacobian (matches official impl)
                new_log_prob = actor.log_prob_via_inverse(
                    mb_obs, mb_aug_actions, params=actor_params,
                )[:, jnp.newaxis]  # (batch_size, 1)

                # PPO clipped surrogate
                rho_s = jnp.exp(new_log_prob - mb_old_log_probs)
                surrogate1 = rho_s * mb_advantages
                surrogate2 = jnp.clip(rho_s, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

                # Entropy loss (maximize entropy = minimize mean log_prob)
                if entropy_coeff > 0:
                    entropy_loss = entropy_coeff * jnp.mean(new_log_prob)
                else:
                    entropy_loss = 0

                # Compress loss on fresh forward samples (L2 norm, matches official)
                if compress_coef > 0:
                    x1_fresh = actor.forward(mb_obs, compress_x0, params=actor_params)
                    z_f = x1_fresh[:, :actor.a_dim]
                    y_f = x1_fresh[:, actor.a_dim:]
                    compress_loss = compress_coef * jnp.mean(
                        jnp.sqrt(jnp.sum((z_f - y_f) ** 2, axis=-1) + 1e-8)
                    )
                else:
                    compress_loss = 0

                total_loss = policy_loss + entropy_loss + compress_loss

                return total_loss, {
                    "loss/policy_loss": policy_loss,
                    "loss/entropy_loss": entropy_loss,
                    "loss/compress_loss": compress_loss,
                    "misc/entropy": -jnp.mean(new_log_prob),
                    "misc/policy_ratio": jnp.mean(rho_s),
                    "misc/clipped_ratio": jnp.mean(jnp.abs(rho_s - 1.0) > clip_epsilon),
                    "misc/log_prob_mean": jnp.mean(new_log_prob),
                }

            new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

            def critic_loss_fn(critic_params, dropout_rng):
                v = critic.apply(
                    {"params": critic_params}, mb_obs,
                    training=True, rngs={"dropout": dropout_rng},
                )
                v_error = (mb_gae_vs - v) * (1 - mb_truncations)
                v_loss = jnp.mean(v_error ** 2)
                return v_loss, {
                    "loss/value_loss": v_loss,
                    "misc/value_mean": jnp.mean(v),
                }

            new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

            metrics = {**actor_metrics, **critic_metrics}
            return (rng, new_actor, new_critic), metrics

        (rng, actor, critic), mb_metrics = jax.lax.scan(
            minibatch_step, init=(rng, actor, critic), xs=mb_indices,
        )
        return (rng, actor, critic), mb_metrics

    (rng, actor, critic), all_metrics = jax.lax.scan(
        epoch_step, init=(rng, actor, critic), length=num_epochs,
    )

    metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
    metrics.update({
        "misc/reward_mean": rollout.rewards.mean(),
        "misc/obs_mean": flat_obs.mean(),
        "misc/obs_std": flat_obs.std(axis=0).mean(),
        "misc/action_l1_mean": jnp.abs(flat_actions).mean(),
        "misc/advantages_mean": flat_advantages.mean(),
        "misc/advantages_std": flat_advantages.std(axis=0).mean(),
    })

    return rng, actor, critic, metrics


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
            optimizer=optax.adam(learning_rate=cfg.flow.lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

        critic_def = ScalarCritic(
            backbone=backbone_cls(
                hidden_dims=cfg.critic_hidden_dims,
                activation=critic_activation,
            ),
        )
        self.critic = Model.create(
            critic_def, critic_rng,
            inputs=(jnp.ones((1, obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

    def train_step(self, rollout: RolloutBatch, step: int) -> Metric:
        self.rng, self.actor, self.critic, metrics = jit_update_genpo(
            self.rng, self.actor, self.critic, rollout,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_epsilon=self.cfg.clip_epsilon,
            entropy_coeff=self.cfg.entropy_coeff,
            compress_coef=self.cfg.compress_coef,
            reward_scaling=self.cfg.reward_scaling,
            normalize_advantage=self.cfg.normalize_advantage,
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
