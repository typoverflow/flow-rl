"""Flow Policy Optimization (FPO) implementation.

Based on: https://arxiv.org/abs/2507.21053
Reference: https://github.com/kvfrans/fpo (playground implementation)

Uses rectified-flow / linear-interpolation flow matching (consistent with flowrl/flow/cnf.py).
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.ppo import compute_gae
from flowrl.config.online.algo.fpo import FPOConfig
from flowrl.flow.cnf import ContinuousNormalizingFlow, FlowBackbone
from flowrl.module.critic import ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import Metric, PRNGKey, RolloutBatch


# ============ Core FPO Functions ============

def compute_cfm_loss(
    actor_params,
    actor: ContinuousNormalizingFlow,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    eps: jnp.ndarray,
    t: jnp.ndarray,
    output_scale: float,
    output_mode: str = "u_but_supervise_as_eps",
) -> jnp.ndarray:
    """Compute CFM loss for given (eps, t) samples.

    Uses rectified-flow / linear-interpolation (consistent with CNF.linear_interpolation):
    - x_t = (1 - t) * eps + t * action
    - velocity_gt = action - eps
    - At t=0: x_t = eps (noise); at t=1: x_t = action

    Args:
        actor_params: Actor parameters (passed explicitly for gradient computation)
        actor: ContinuousNormalizingFlow model (for structure only)
        obs: (B, obs_dim) normalized observations
        actions: (B, action_dim) sampled actions
        eps: (B, N_mc, action_dim) noise samples
        t: (B, N_mc, 1) timestep samples
        output_scale: Scale factor for network output
        output_mode: "u" (velocity MSE) or "u_but_supervise_as_eps" (epsilon prediction)

    Returns:
        (B, N_mc) CFM loss per sample
    """
    B, N_mc = eps.shape[:2]
    action_dim = actions.shape[-1]
    obs_dim = obs.shape[-1]

    # Expand obs and actions to match N_mc
    obs_exp = jnp.broadcast_to(obs[:, None, :], (B, N_mc, obs_dim))
    actions_exp = jnp.broadcast_to(actions[:, None, :], (B, N_mc, action_dim))

    # Rectified-flow interpolation: x_t = (1-t)*eps + t*action
    x_t = (1.0 - t) * eps + t * actions_exp

    # Flatten for network forward pass
    x_t_flat = x_t.reshape(B * N_mc, action_dim)
    obs_flat = obs_exp.reshape(B * N_mc, obs_dim)
    t_flat = t.reshape(B * N_mc, 1)

    # Predict velocity using actor.apply with explicit params (training=False for deterministic)
    vel_pred = actor.apply(
        {"params": actor_params},
        x_t_flat,
        t_flat,
        condition=obs_flat,
        training=False,
    )
    vel_pred = vel_pred * output_scale
    vel_pred = vel_pred.reshape(B, N_mc, action_dim)

    if output_mode == "u":
        # Velocity MSE: supervise velocity directly
        velocity_gt = actions_exp - eps
        loss = jnp.mean((vel_pred - velocity_gt) ** 2, axis=-1)
    else:  # "u_but_supervise_as_eps" (recommended by paper)
        # Epsilon prediction: convert velocity pred to x1 pred, then MSE against eps
        # x_t = (1-t)*x0 + t*x1, velocity = x1 - x0
        # x0_pred = x_t - t * velocity_pred
        # x1_pred = x0_pred + velocity_pred = x_t + (1-t) * velocity_pred
        # In our convention: x0=eps (noise), x1=action
        # So we predict x0 (eps) from velocity
        x0_pred = x_t - t * vel_pred  # predicted noise
        loss = jnp.mean((eps - x0_pred) ** 2, axis=-1)

    return loss


@partial(jax.jit, static_argnames=(
    "gamma", "gae_lambda", "clip_epsilon",
    "reward_scaling", "normalize_advantage",
    "num_epochs", "num_minibatches", "batch_size",
    "value_loss_coeff", "output_scale", "log_ratio_clip", "output_mode",
    "average_losses_before_exp",
))
def jit_update_fpo(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    critic: Model,
    rollout: RolloutBatch,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    reward_scaling: float,
    normalize_advantage: bool,
    num_epochs: int,
    num_minibatches: int,
    batch_size: int,
    value_loss_coeff: float,
    output_scale: float,
    log_ratio_clip: float,
    output_mode: str,
    average_losses_before_exp: bool,
):
    """FPO training update.

    Key difference from PPO: ratio = exp(mean(L_old) - mean(L_new))
    where L is the CFM loss computed with shared (eps, t) samples.
    """
    T, B = rollout.rewards.shape[:2]

    # ===== Compute value predictions and GAE (same as PPO) =====
    value_pred = critic(rollout.obs)
    next_value_pred = critic(rollout.next_obs)

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

    if normalize_advantage:
        gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)

    # ===== Flatten rollout data =====
    flat_obs = rollout.obs.reshape(T * B, -1)
    flat_actions = rollout.actions.reshape(T * B, -1)
    flat_advantages = gae_advantages.reshape(T * B, 1)
    flat_gae_vs = gae_vs.reshape(T * B, 1)
    flat_truncations = rollout.truncated.reshape(T * B, 1)

    # Flow data: (T, B, N_mc, ...) -> (T*B, N_mc, ...)
    N_mc = rollout.flow_noise.shape[2]
    flat_noise = rollout.flow_noise.reshape(T * B, N_mc, -1)
    flat_timesteps = rollout.flow_timesteps.reshape(T * B, N_mc, 1)
    flat_initial_cfm_loss = rollout.initial_cfm_loss.reshape(T * B, N_mc)

    # ===== Epoch/minibatch loop =====
    def epoch_step(carry, _):
        rng, actor, critic = carry
        rng, perm_rng = jax.random.split(rng)

        perm = jax.random.permutation(perm_rng, T * B)
        total = num_minibatches * batch_size
        perm = perm[:total]
        mb_indices = perm.reshape(num_minibatches, batch_size)

        def minibatch_step(carry, indices):
            rng, actor, critic = carry

            mb_obs = flat_obs[indices]
            mb_actions = flat_actions[indices]
            mb_noise = flat_noise[indices]
            mb_timesteps = flat_timesteps[indices]
            mb_advantages = flat_advantages[indices]
            mb_gae_vs = flat_gae_vs[indices]
            mb_truncations = flat_truncations[indices]
            mb_initial_cfm_loss = flat_initial_cfm_loss[indices]

            # ===== Actor loss: FPO ratio =====
            def actor_loss_fn(actor_params, dropout_rng):
                new_cfm_loss = compute_cfm_loss(
                    actor_params=actor_params,
                    actor=actor,
                    obs=mb_obs,
                    actions=mb_actions,
                    eps=mb_noise,
                    t=mb_timesteps,
                    output_scale=output_scale,
                    output_mode=output_mode,
                )

                if average_losses_before_exp:
                    # Paper's estimator: exp(mean(L_old) - mean(L_new))
                    # Average over MC samples BEFORE exponentiating
                    log_ratio = (
                        jnp.mean(mb_initial_cfm_loss, axis=-1, keepdims=True) -
                        jnp.mean(new_cfm_loss, axis=-1, keepdims=True)
                    )
                    # Clip log-ratio to prevent exp overflow
                    log_ratio_clipped = jnp.clip(log_ratio, -log_ratio_clip, log_ratio_clip)
                    ratio = jnp.exp(log_ratio_clipped)
                else:
                    # Alternative: clip per-sample, then exp, then average
                    # ratio = mean(exp(clip(L_old - L_new)))
                    per_sample_log_ratio = mb_initial_cfm_loss - new_cfm_loss  # (B, N_mc)
                    per_sample_log_ratio_clipped = jnp.clip(
                        per_sample_log_ratio, -log_ratio_clip, log_ratio_clip
                    )
                    ratio = jnp.mean(jnp.exp(per_sample_log_ratio_clipped), axis=-1, keepdims=True)
                    log_ratio = jnp.mean(per_sample_log_ratio, axis=-1, keepdims=True)
                    log_ratio_clipped = jnp.mean(per_sample_log_ratio_clipped, axis=-1, keepdims=True)

                # PPO surrogate with FPO ratio
                surrogate1 = ratio * mb_advantages
                surrogate2 = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

                return policy_loss, {
                    "loss/policy_loss": policy_loss,
                    "misc/fpo_ratio_mean": jnp.mean(ratio),
                    "misc/fpo_ratio_min": jnp.min(ratio),
                    "misc/fpo_ratio_max": jnp.max(ratio),
                    "misc/clipped_ratio_frac": jnp.mean(jnp.abs(ratio - 1.0) > clip_epsilon),
                    "misc/cfm_loss_mean": jnp.mean(new_cfm_loss),
                    "misc/cfm_loss_std": jnp.std(new_cfm_loss),
                    "misc/log_ratio_mean": jnp.mean(log_ratio),
                    "misc/log_ratio_std": jnp.std(log_ratio),
                    "misc/log_ratio_clipped_frac": jnp.mean(
                        (log_ratio >= log_ratio_clip) | (log_ratio <= -log_ratio_clip)
                    ),
                }

            new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

            # ===== Critic loss =====
            def critic_loss_fn(critic_params, dropout_rng):
                v = critic.apply(
                    {"params": critic_params},
                    mb_obs,
                    training=True,
                    rngs={"dropout": dropout_rng},
                )
                v_error = (mb_gae_vs - v) * (1 - mb_truncations)
                v_loss = jnp.mean(v_error ** 2) * value_loss_coeff
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
    metrics["misc/reward_mean"] = rollout.rewards.mean()
    metrics["misc/advantages_mean"] = gae_advantages.mean()
    metrics["misc/advantages_std"] = gae_advantages.std()

    return rng, actor, critic, metrics


@partial(jax.jit, static_argnames=("deterministic", "n_mc", "output_scale", "output_mode"))
def jit_sample_action_fpo(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    obs: jnp.ndarray,
    deterministic: bool,
    n_mc: int,
    output_scale: float,
    output_mode: str,
):
    """Sample action from flow policy and compute initial CFM loss.

    Uses CNF.sample() for action generation with continuous t sampling.
    """
    # Properly split RNG for different uses
    rng, x0_rng, solver_rng, eps_rng, t_rng = jax.random.split(rng, 5)

    B = obs.shape[0]
    act_dim = actor.x_dim

    # ===== Sample action using CNF =====
    # For deterministic: use x0_rng (consistent per-call) but no solver noise
    # For stochastic: use x0_rng for initial noise
    x0 = jax.random.normal(x0_rng, (B, act_dim))

    # CNF.sample() handles Euler integration
    _, action, _ = actor.sample(
        rng=solver_rng,
        x0=x0,
        condition=obs,
        training=False,
    )

    # ===== Sample (eps, t) for CFM loss =====
    eps = jax.random.normal(eps_rng, (B, n_mc, act_dim))
    # Continuous t sampling (matches CNF.linear_interpolation)
    t = jax.random.uniform(t_rng, (B, n_mc, 1))

    # ===== Compute initial CFM loss =====
    initial_cfm_loss = compute_cfm_loss(
        actor_params=actor.state.params,
        actor=actor,
        obs=obs,
        actions=action,
        eps=eps,
        t=t,
        output_scale=output_scale,
        output_mode=output_mode,
    )

    return action, eps, t, initial_cfm_loss


class FPOAgent(BaseAgent):
    """Flow Policy Optimization (FPO) Agent.

    Uses ContinuousNormalizingFlow with rectified-flow / linear-interpolation.
    Implements the PPO-level FPO estimator: ratio = exp(mean(L_old) - mean(L_new)).
    """
    name = "FPOAgent"
    model_names = ["actor", "critic"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: FPOConfig, seed: int):
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

        # ===== Critic (same as PPO) =====
        critic_def = ScalarCritic(
            backbone=backbone_cls(
                hidden_dims=cfg.critic_hidden_dims,
                activation=activation,
            ),
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

        # ===== Flow Actor using ContinuousNormalizingFlow =====
        flow_backbone = FlowBackbone(
            vel_predictor=backbone_cls(
                hidden_dims=cfg.flow_hidden_dims,
                output_dim=self.act_dim,
                activation=activation,
            ),
            time_embedding=PositionalEmbedding(output_dim=cfg.timestep_embed_dim),
            cond_embedding=None,
        )
        self.actor = ContinuousNormalizingFlow.create(
            network=flow_backbone,
            rng=actor_rng,
            inputs=(
                jnp.ones((1, self.act_dim)),   # x
                jnp.ones((1, 1)),              # time
                jnp.ones((1, self.obs_dim)),   # condition
            ),
            x_dim=self.act_dim,
            steps=cfg.flow_steps,
            clip_sampler=True,
            x_min=-1.0,
            x_max=1.0,
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

    def train_step(self, rollout: RolloutBatch, step: int) -> Metric:
        self.rng, self.actor, self.critic, metrics = jit_update_fpo(
            self.rng,
            self.actor,
            self.critic,
            rollout,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_epsilon=self.cfg.clip_epsilon,
            reward_scaling=self.cfg.reward_scaling,
            normalize_advantage=self.cfg.normalize_advantage,
            num_epochs=self.cfg.num_epochs,
            num_minibatches=self.cfg.num_minibatches,
            batch_size=self.cfg.batch_size,
            value_loss_coeff=self.cfg.value_loss_coeff,
            output_scale=self.cfg.policy_output_scale,
            log_ratio_clip=self.cfg.log_ratio_clip,
            output_mode=self.cfg.output_mode,
            average_losses_before_exp=self.cfg.average_losses_before_exp,
        )
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        assert num_samples == 1, "FPO only supports num_samples=1"
        self.rng, sample_key = jax.random.split(self.rng)

        action, eps, t, initial_cfm_loss = jit_sample_action_fpo(
            sample_key,
            self.actor,
            obs,
            deterministic,
            n_mc=self.cfg.n_mc,
            output_scale=self.cfg.policy_output_scale,
            output_mode=self.cfg.output_mode,
        )

        return action, {
            "flow_noise": eps,
            "flow_timesteps": t,
            "initial_cfm_loss": initial_cfm_loss,
        }
