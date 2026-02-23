from functools import partial

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.ppo import PPOConfig
from flowrl.module.actor import TanhMeanGaussianActor
from flowrl.module.critic import ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.types import Metric, PRNGKey, RolloutBatch


def compute_gae(
    terminated: jnp.ndarray,   # (T, B, 1)
    truncated: jnp.ndarray,     # (T, B, 1)
    rewards: jnp.ndarray,      # (T, B, 1)
    values: jnp.ndarray,       # (T, B, 1)
    next_values: jnp.ndarray,  # (T, B, 1)
    gae_lambda: float,
    gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    # TD residual: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    deltas = rewards + gamma * (1 - terminated) * next_values - values

    # GAE recursion: A_t = δ_t + (γλ) * A_{t+1}
    episode_ended = jnp.maximum(terminated, truncated)
    gae_discount = (1.0 - episode_ended) * gamma * gae_lambda

    def gae_step(carry: jnp.ndarray, x: tuple) -> tuple[jnp.ndarray, jnp.ndarray]:
        delta_t, disc_t = x
        a_next = carry
        a_t = delta_t + disc_t * a_next
        return a_t, a_t

    _, advantages = jax.lax.scan(
        gae_step,
        init=jnp.zeros_like(values[0]),
        xs=(deltas[::-1], gae_discount[::-1]),
    )
    advantages = advantages[::-1]  # [a_0, ..., a_{T-1}]

    # Lambda-return (critic target): V_target = A + V
    lambda_returns = advantages + values

    return lambda_returns, advantages



@partial(jax.jit, static_argnames=(
    "gamma", "gae_lambda", "clip_epsilon", "entropy_coeff",
    "reward_scaling", "normalize_advantage",
    "num_epochs", "num_minibatches", "batch_size",
))
def jit_update_ppo(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    rollout: RolloutBatch,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    entropy_coeff: float,
    reward_scaling: float,
    normalize_advantage: bool,
    num_epochs: int,
    num_minibatches: int,
    batch_size: int,
):
    T, B = rollout.rewards.shape[:2]

    # Compute value predictions for all obs
    value_pred = critic(rollout.obs)  # (T, B, 1)
    next_value_pred = critic(rollout.next_obs)

    # Compute GAE
    gae_vs, gae_advantages = jax.lax.stop_gradient(
        compute_gae(
            terminated=rollout.terminated,
            truncated=rollout.truncated,
            rewards=rollout.rewards * reward_scaling,
            values=value_pred,
            next_values=next_value_pred,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )
    )

    # Normalize advantages
    if normalize_advantage:
        gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)

    # Flatten rollout data: (T, B, ...) -> (T*B, ...)
    flat_obs = rollout.obs.reshape(T * B, -1)
    flat_actions = rollout.actions.reshape(T * B, -1)
    flat_old_log_probs = rollout.log_probs.reshape(T * B, 1)
    flat_advantages = gae_advantages.reshape(T * B, 1)
    flat_gae_vs = gae_vs.reshape(T * B, 1)
    flat_truncations = rollout.truncated.reshape(T * B, 1)

    def epoch_step(carry, _):
        rng, actor, critic = carry
        rng, perm_rng = jax.random.split(rng)

        # Shuffle and create minibatches
        perm = jax.random.permutation(perm_rng, T * B)
        # Truncate to num_minibatches * batch_size
        total = num_minibatches * batch_size
        perm = perm[:total]
        mb_indices = perm.reshape(num_minibatches, batch_size)

        def minibatch_step(carry, indices):
            rng, actor, critic = carry
            rng, entropy_rng = jax.random.split(rng)

            mb_obs = flat_obs[indices]
            mb_actions = flat_actions[indices]
            mb_old_log_probs = flat_old_log_probs[indices]
            mb_advantages = flat_advantages[indices]
            mb_gae_vs = flat_gae_vs[indices]
            mb_truncations = flat_truncations[indices]

            # Joint loss over actor and critic
            def actor_loss_fn(actor_params, dropout_rng):
                action_dist = actor.apply(
                    {"params": actor_params},
                    mb_obs,
                    training=True,
                    rngs={"dropout": dropout_rng},
                )
                # MultivariateNormalDiag.log_prob already sums across dims
                new_log_probs = action_dist.log_prob(mb_actions)[..., jnp.newaxis]

                rho_s = jnp.exp(new_log_probs - mb_old_log_probs)
                surrogate1 = rho_s * mb_advantages
                surrogate2 = jnp.clip(rho_s, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

                entropy = action_dist.entropy()
                entropy = jnp.mean(entropy)
                entropy_loss = -entropy_coeff * entropy

                actor_total = policy_loss + entropy_loss
                return actor_total, {
                    "loss/policy_loss": policy_loss,
                    "loss/entropy_loss": entropy_loss,
                    "misc/entropy": entropy,
                    "misc/policy_ratio": jnp.mean(rho_s),
                    "misc/clipped_ratio": jnp.mean(jnp.abs(rho_s - 1.0) > clip_epsilon),
                    "misc/action_mean": jnp.mean(action_dist.mean()),
                }

            new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

            def critic_loss_fn(critic_params, dropout_rng):
                v = critic.apply(
                    {"params": critic_params},
                    mb_obs,
                    training=True,
                    rngs={"dropout": dropout_rng},
                )
                v_error = mb_gae_vs - v
                v_loss = jnp.mean(v_error ** 2)
                return v_loss, {
                    "loss/value_loss": v_loss,
                    "misc/value_mean": jnp.mean(v),
                }

            new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

            metrics = {**actor_metrics, **critic_metrics}
            return (rng, new_actor, new_critic), metrics

        (rng, actor, critic), mb_metrics = jax.lax.scan(
            minibatch_step,
            init=(rng, actor, critic),
            xs=mb_indices,
        )
        return (rng, actor, critic), mb_metrics

    (rng, actor, critic), all_metrics = jax.lax.scan(
        epoch_step,
        init=(rng, actor, critic),
        length=num_epochs,
    )

    # Average metrics across epochs and minibatches
    metrics = jax.tree.map(lambda x: x.mean(), all_metrics)

    # Add advantage stats (pre-normalization)
    metrics["misc/reward_mean"] = rollout.rewards.mean()
    metrics["misc/advantages_mean"] = gae_advantages.mean()
    metrics["misc/advantages_std"] = gae_advantages.std()

    return rng, actor, critic, metrics


@partial(jax.jit, static_argnames=("deterministic",))
def jit_sample_action(
    rng: PRNGKey,
    actor: Model,
    obs: jnp.ndarray,
    deterministic: bool,
):
    dist = actor(obs, training=False)
    if deterministic:
        action = dist.mean()
        log_prob = jnp.zeros(obs.shape[0])
    else:
        action = dist.sample(seed=rng)
        log_prob = dist.log_prob(action)
    return action, log_prob


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO)"""
    name = "PPOAgent"
    model_names = ["actor", "critic"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        activation = {
            "relu": jax.nn.relu,
            "elu": jax.nn.elu,
            "silu": jax.nn.silu,
        }[cfg.activation]
        backbone_cls = {
            "mlp": MLP,
            "simba": Simba,
        }[cfg.backbone_cls]

        actor_def = TanhMeanGaussianActor(
            backbone=backbone_cls(
                hidden_dims=cfg.actor_hidden_dims,
                activation=activation,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            conditional_logstd=True,
            logstd_min=-10,
        )
        critic_def = ScalarCritic(
            backbone=backbone_cls(
                hidden_dims=cfg.critic_hidden_dims,
                activation=activation,
            ),
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
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

    def train_step(self, rollout: RolloutBatch, step: int) -> Metric:
        self.rng, self.actor, self.critic, metrics = jit_update_ppo(
            self.rng,
            self.actor,
            self.critic,
            rollout,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_epsilon=self.cfg.clip_epsilon,
            entropy_coeff=self.cfg.entropy_coeff,
            reward_scaling=self.cfg.reward_scaling,
            normalize_advantage=self.cfg.normalize_advantage,
            num_epochs=self.cfg.num_epochs,
            num_minibatches=self.cfg.num_minibatches,
            batch_size=self.cfg.batch_size,
        )
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        assert num_samples == 1, "PPO only supports num_samples=1"
        self.rng, sample_key = jax.random.split(self.rng)
        action, log_prob = jit_sample_action(
            sample_key,
            self.actor,
            obs,
            deterministic,
        )
        return action, {"log_prob": log_prob[..., jnp.newaxis]}
