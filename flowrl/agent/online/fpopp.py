from functools import partial

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.fpo import FPOAgent, OutputScale, compute_cfm_loss
from flowrl.agent.online.ppo import compute_gae
from flowrl.config.online.algo.fpopp import FPOPPConfig
from flowrl.flow.cnf import ContinuousNormalizingFlow, FlowBackbone
from flowrl.functional.activation import get_activation
from flowrl.module.critic import ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Metric, PRNGKey, RolloutBatch


def clamp_ste(x, min_val, max_val):
    """Clamp with straight-through estimator: forward uses clamped value, backward passes gradients through."""
    clamped = jnp.clip(x, a_min=min_val, a_max=max_val)
    return x + jax.lax.stop_gradient(clamped - x)


@partial(jax.jit, static_argnames=(
    "gamma", "gae_lambda", "clip_epsilon",
    "reward_scaling", "normalize_advantage",
    "num_epochs", "num_minibatches", "batch_size",
    "num_mc_samples",
    "cfm_loss_clamp",
    "cfm_loss_clamp_negative_advantages_max",
    "cfm_diff_clamp_max",
    "advantage_clamp",
))
def jit_update_fpopp(
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
    num_mc_samples: int,
    cfm_loss_clamp: float,
    cfm_loss_clamp_negative_advantages_max: float,
    cfm_diff_clamp_max: float,
    advantage_clamp: float,
):
    T, B = rollout.rewards.shape[:2]

    # Compute value predictions for all obs
    value_pred = critic(rollout.obs)
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

    if normalize_advantage:
        gae_advantages = (gae_advantages - gae_advantages.mean()) / (gae_advantages.std() + 1e-8)

    # ===== Flatten rollout data =====
    flat_obs = rollout.obs.reshape(T * B, -1)
    flat_actions = rollout.actions.reshape(T * B, -1)
    flat_advantages = gae_advantages.reshape(T * B, 1)
    flat_gae_vs = gae_vs.reshape(T * B, 1)
    flat_truncations = rollout.truncated.reshape(T * B, 1)

    flat_cfm_loss = rollout.extras["cfm_loss"].reshape(T * B, num_mc_samples, 1)
    flat_eps = rollout.extras["eps"].reshape(T * B, num_mc_samples, -1)
    flat_t = rollout.extras["t"].reshape(T * B, num_mc_samples, 1)

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
            mb_advantages = flat_advantages[indices]
            mb_gae_vs = flat_gae_vs[indices]
            mb_truncations = flat_truncations[indices]

            mb_cfm_loss = flat_cfm_loss[indices]
            mb_eps = flat_eps[indices]
            mb_t = flat_t[indices]

            # Clamp advantages
            if advantage_clamp > 0:
                mb_advantages = jnp.clip(mb_advantages, -advantage_clamp, advantage_clamp)

            # ===== Actor loss: FPO++ per-sample ratio + ASPO =====
            def actor_loss_fn(actor_params, dropout_rng):
                new_cfm_loss, vel_pred = compute_cfm_loss(
                    params=actor_params,
                    actor=actor,
                    obs=mb_obs,
                    action=mb_actions,
                    eps=mb_eps,
                    t=mb_t,
                    num_mc_samples=num_mc_samples,
                    output_mode="u",
                )

                old_cfm = mb_cfm_loss

                # Symmetric CFM loss clamping
                if cfm_loss_clamp > 0:
                    old_cfm = jnp.clip(old_cfm, a_max=cfm_loss_clamp)
                    new_cfm_loss = jnp.clip(new_cfm_loss, a_max=cfm_loss_clamp)

                # Clamp new CFM loss for negative advantages to prevent unbounded growth
                adv = mb_advantages[:, None, :]  # (B, 1, 1)
                if cfm_loss_clamp_negative_advantages_max > 0:
                    new_cfm_loss = jnp.where(
                        adv < 0,
                        jnp.clip(new_cfm_loss, a_max=cfm_loss_clamp_negative_advantages_max),
                        new_cfm_loss,
                    )

                # STE log-ratio clamping
                log_ratio = old_cfm - new_cfm_loss  # (B, num_mc_samples, 1)
                log_ratio = clamp_ste(log_ratio, min_val=-cfm_diff_clamp_max, max_val=cfm_diff_clamp_max)
                rho_s = jnp.exp(log_ratio)

                # Positive advantages: standard PPO clipping (Eq 1)
                ppo_surr1 = rho_s * adv
                ppo_surr2 = jnp.clip(rho_s, 1 - clip_epsilon, 1 + clip_epsilon) * adv
                ppo_obj = jnp.minimum(ppo_surr1, ppo_surr2)

                # Negative advantages: SPO objective (Eq 11)
                spo_obj = rho_s * adv - jnp.abs(adv) / (2 * clip_epsilon) * (rho_s - 1) ** 2

                # Asymmetric selection (Eq 12)
                aspo_obj = jnp.where(adv >= 0, ppo_obj, spo_obj)

                # mean over batch (Eq 13)
                policy_loss = -jnp.mean(aspo_obj)

                return policy_loss, {
                    "loss/policy_loss": policy_loss,
                    "misc/policy_ratio": jnp.mean(rho_s),
                    "misc/clipped_ratio": jnp.mean(jnp.abs(rho_s - 1.0) > clip_epsilon),
                    "misc/cfm_loss_mean": jnp.mean(new_cfm_loss),
                    "misc/cfm_loss_gap": jnp.mean(jnp.abs(log_ratio)),
                    "misc/vel_pred_l1_mean": jnp.abs(vel_pred).mean(),
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


@partial(jax.jit, static_argnames=("deterministic", "num_mc_samples", "additive_noise"))
def jit_sample_action_fpopp(
    rng: PRNGKey,
    actor: ContinuousNormalizingFlow,
    obs: jnp.ndarray,
    deterministic: bool,
    additive_noise: float,
    num_mc_samples: int,
):
    """Sample action from flow policy with zero-sampling for evaluation.

    When deterministic=True, uses x0=0 instead of x0~N(0,I) for improved
    evaluation performance (Section 4.1 of FPO++ paper).
    """
    rng, x0_rng, noise_rng, solver_rng, eps_rng, t_rng = jax.random.split(rng, 6)

    B = obs.shape[0]
    x0_random = jax.random.normal(x0_rng, (B, actor.x_dim))
    x0_zero = jnp.zeros((B, actor.x_dim))
    x0 = jax.lax.cond(deterministic, lambda: x0_zero, lambda: x0_random)

    _, action, _ = actor.sample(
        rng=solver_rng,
        x0=x0,
        condition=obs,
        training=False,
    )

    if not deterministic:
        action = action + additive_noise * jax.random.normal(noise_rng, (B, actor.x_dim))

    eps = jax.random.normal(eps_rng, (B, num_mc_samples, actor.x_dim))
    # Sample from [1, steps) to avoid t=0 boundary instabilities
    t_idx = jax.random.randint(t_rng, (B, num_mc_samples, 1), 1, actor.steps)
    t = actor.step2t(t_idx)

    loss, vel_pred = compute_cfm_loss(
        params=actor.state.params,
        actor=actor,
        obs=obs,
        action=action,
        eps=eps,
        t=t,
        num_mc_samples=num_mc_samples,
        output_mode="u",
    )

    return action, loss, eps, t


class FPOPPAgent(FPOAgent):
    """
    Flow Policy Optimization++ (FPO++)
    https://arxiv.org/abs/2602.02481
    """
    name = "FPOPPAgent"

    def __init__(self, obs_dim: int, act_dim: int, cfg: FPOPPConfig, seed: int):
        # Skip FPOAgent.__init__ to override optimizers with paper-specified
        # Adam betas=(0.9, 0.95) and weight_decay=1e-4 (Table A.1)
        BaseAgent.__init__(self, obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        critic_activation = get_activation(cfg.critic_activation)
        actor_activation = get_activation(cfg.flow.activation)
        backbone_cls = {
            "mlp": MLP,
            "simba": Simba,
        }[cfg.backbone_cls]

        critic_def = ScalarCritic(
            backbone=backbone_cls(
                hidden_dims=cfg.critic_hidden_dims,
                activation=critic_activation,
            ),
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adamw(
                learning_rate=cfg.critic_lr, b1=0.9, b2=0.95, weight_decay=1e-4,
            ),
            clip_grad_norm=cfg.clip_grad_norm,
        )

        flow_backbone = OutputScale(
            backbone=FlowBackbone(
                vel_predictor=backbone_cls(
                    hidden_dims=cfg.flow.hidden_dims,
                    activation=actor_activation,
                    output_dim=self.act_dim,
                ),
                time_embedding=LearnableFourierEmbedding(output_dim=cfg.flow.time_dim),
            ),
            output_scale=cfg.flow.output_scale,
        )
        self.actor = ContinuousNormalizingFlow.create(
            network=flow_backbone,
            rng=actor_rng,
            inputs=(
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, 1)),
                jnp.ones((1, self.obs_dim)),
            ),
            x_dim=self.act_dim,
            steps=cfg.flow.steps,
            clip_sampler=cfg.flow.clip_sampler,
            x_min=cfg.flow.x_min,
            x_max=cfg.flow.x_max,
            optimizer=optax.adamw(
                learning_rate=cfg.flow.lr, b1=0.9, b2=0.95, weight_decay=1e-4,
            ),
            clip_grad_norm=cfg.clip_grad_norm,
        )

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        self.rng, sample_rng = jax.random.split(self.rng)

        action, loss, eps, t = jit_sample_action_fpopp(
            sample_rng,
            self.actor,
            obs,
            deterministic,
            additive_noise=self.cfg.additive_noise,
            num_mc_samples=self.cfg.num_mc_samples,
        )

        return action, {
            "cfm_loss": loss,
            "eps": eps,
            "t": t,
        }

    def train_step(self, rollout: RolloutBatch, step: int) -> Metric:
        self.rng, self.actor, self.critic, metrics = jit_update_fpopp(
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
            num_mc_samples=self.cfg.num_mc_samples,
            cfm_loss_clamp=self.cfg.cfm_loss_clamp,
            cfm_loss_clamp_negative_advantages_max=self.cfg.cfm_loss_clamp_negative_advantages_max,
            cfm_diff_clamp_max=self.cfg.cfm_diff_clamp_max,
            advantage_clamp=self.cfg.advantage_clamp,
        )
        return metrics
