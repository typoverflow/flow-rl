from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.agent.online.ppo import compute_gae
from flowrl.config.online.algo.dppo import DPPOConfig
from flowrl.flow.continuous_ddpm import (
    ContinuousDDPM,
    ContinuousDDPMBackbone,
    quad_t_schedule,
)
from flowrl.functional.activation import get_activation
from flowrl.module.critic import ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.module.time_embedding import LearnableFourierEmbedding
from flowrl.types import Metric, PRNGKey, RolloutBatch


@partial(jax.jit, static_argnames=("steps", "min_logprob_std"))
def jit_compute_chain_log_probs(
    actor: ContinuousDDPM, 
    obs: jnp.ndarray, 
    chain: jnp.ndarray,
    steps: int, 
    min_logprob_std: float,
) -> jnp.ndarray:
    ts = quad_t_schedule(steps, n=actor.t_schedule_n,
                         tmin=actor.t_diffusion[0], tmax=actor.t_diffusion[1])
    alpha_hats = actor.noise_schedule_func(ts)[0] ** 2
    alphas = alpha_hats[1:] / alpha_hats[:-1]
    alphas = jnp.concat([jnp.ones((1,)), alphas], axis=0)
    betas = 1 - alphas

    N = obs.shape[0]
    t_proto = jnp.ones((N, 1), dtype=jnp.float32)

    def step_fn(_, i):
        idx = steps - i
        xt, xt_1 = chain[:, idx], chain[:, idx + 1]
        eps_theta = actor(xt, t_proto * ts[i], condition=obs, training=False)
        # reverse mean
        x0_hat = (xt - jnp.sqrt(1 - alpha_hats[i]) * eps_theta) / jnp.sqrt(alpha_hats[i])
        c1 = jnp.sqrt(alpha_hats[i-1]) * betas[i] / (1 - alpha_hats[i])
        c2 = jnp.sqrt(alphas[i]) * (1 - alpha_hats[i-1]) / (1 - alpha_hats[i])
        mean = c1 * x0_hat + c2 * xt
        std = jnp.maximum(jnp.sqrt(betas[i]), min_logprob_std)
        # gaussian log prob
        d = xt_1.shape[-1]
        lp = -0.5 * jnp.sum(((xt_1 - mean) / std) ** 2, axis=-1)
        lp = lp - 0.5 * d * jnp.log(2 * jnp.pi) - d * jnp.log(std)
        return _, jnp.clip(lp, -5.0, 2.0)

    _, step_lps = jax.lax.scan(step_fn, None, jnp.arange(steps, 0, -1))
    return jnp.transpose(step_lps, (1, 0))


@partial(jax.jit, static_argnames=("steps", "min_logprob_std"))
def jit_sample_actions(
    rng: PRNGKey, actor: ContinuousDDPM, obs: jnp.ndarray,
    steps: int, min_logprob_std: float,
) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)
    xT = jax.random.normal(xT_rng, (B, actor.x_dim))
    rng, action, history = actor.sample(
        rng, xT, condition=obs, training=False, solver="ddpm",
    )
    chain = jnp.transpose(
        jnp.concatenate([history[0], action[jnp.newaxis]], axis=0), (1, 0, 2))
    step_lps = jit_compute_chain_log_probs(actor, obs, chain, steps, min_logprob_std)
    log_prob = step_lps.mean(axis=-1, keepdims=True)  # not used in DPPO training
    return rng, action, chain, log_prob


@partial(jax.jit, static_argnames=(
    "gamma", 
    "gae_lambda", 
    "gamma_denoising",
    "clip_epsilon", 
    "clip_epsilon_base", 
    "clip_epsilon_rate",
    "reward_scaling", 
    "normalize_advantage",
    "num_epochs", 
    "num_minibatches", 
    "batch_size",
    "denoising_steps", 
    "min_logprob_std",
))
def jit_update_dppo(
    rng: PRNGKey, 
    actor: ContinuousDDPM, 
    critic: Model, 
    rollout: RolloutBatch,
    gamma: float, 
    gae_lambda: float, 
    gamma_denoising: float,
    clip_epsilon: float, 
    clip_epsilon_base: float, 
    clip_epsilon_rate: float,
    reward_scaling: float, 
    normalize_advantage: bool,
    num_epochs: int, 
    num_minibatches: int, 
    batch_size: int,
    denoising_steps: int, 
    min_logprob_std: float,
):
    T, B = rollout.rewards.shape[:2]
    K = denoising_steps

    value_pred = critic(rollout.obs)
    next_value_pred = critic(rollout.next_obs)
    gae_vs, gae_advantages = jax.lax.stop_gradient(compute_gae(
        terminated=rollout.terminated, truncated=rollout.truncated,
        rewards=rollout.rewards * reward_scaling,
        values=value_pred, next_values=next_value_pred,
        gae_lambda=gae_lambda, gamma=gamma,
    ))

    flat_obs = rollout.obs.reshape(T * B, -1)
    flat_chains = rollout.extras["action_chains"].reshape(T * B, K + 1, -1)
    flat_adv = gae_advantages.reshape(T * B, 1)
    flat_vs = gae_vs.reshape(T * B, 1)

    flat_old_step_lps = jax.lax.stop_gradient(
        jit_compute_chain_log_probs(actor, flat_obs, flat_chains, K, min_logprob_std)
    )

    denoising_discounts = jnp.array([gamma_denoising ** (K - 1 - k) for k in range(K)])
    # t_frac = jnp.linspace(0.0, 1.0, K)
    # clip_schedule = clip_epsilon_base + (clip_epsilon - clip_epsilon_base) * (
    #     jnp.where(K > 1,
    #               (jnp.exp(clip_epsilon_rate * t_frac) - 1) / (jnp.exp(clip_epsilon_rate) - 1),
    #               1.0)
    # )
    clip_schedule = jnp.ones((K,), dtype=jnp.float32) * clip_epsilon

    def epoch_step(carry, _):
        rng, actor, critic = carry
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, T * B)[:num_minibatches * batch_size]
        mb_indices = perm.reshape(num_minibatches, batch_size)

        def minibatch_step(carry, indices):
            rng, actor, critic = carry
            rng, _ = jax.random.split(rng)

            mb_obs = flat_obs[indices]
            mb_chains = flat_chains[indices]
            mb_old_step_lps = flat_old_step_lps[indices]
            mb_adv = flat_adv[indices]
            mb_vs = flat_vs[indices]

            if normalize_advantage:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            def actor_loss_fn(actor_params, dropout_rng):
                diff_actor = actor.replace(state=actor.state.replace(params=actor_params))
                new_step_lps = jit_compute_chain_log_probs(
                    diff_actor, mb_obs, mb_chains, K, min_logprob_std)

                ratios = jnp.exp(new_step_lps - mb_old_step_lps)
                weighted_adv = mb_adv * denoising_discounts[jnp.newaxis, :]
                s1 = ratios * weighted_adv
                s2 = jnp.clip(ratios, 1 - clip_schedule[jnp.newaxis, :],
                               1 + clip_schedule[jnp.newaxis, :]) * weighted_adv
                loss = -jnp.mean(jnp.minimum(s1, s2))
                return loss, {
                    "loss/policy_loss": loss,
                    "misc/policy_ratio": jnp.mean(ratios),
                }

            new_actor, actor_info = actor.apply_gradient(actor_loss_fn)

            def critic_loss_fn(critic_params, dropout_rng):
                v = critic.apply({"params": critic_params}, 
                                 mb_obs,
                                 training=True, 
                                 rngs={"dropout": dropout_rng})
                v_loss = jnp.mean((mb_vs - v) ** 2)
                return v_loss, {
                    "loss/value_loss": v_loss, 
                    "misc/value_mean": jnp.mean(v)}

            new_critic, critic_info = critic.apply_gradient(critic_loss_fn)
            return (rng, new_actor, new_critic), {**actor_info, **critic_info}

        (rng, actor, critic), mb_metrics = jax.lax.scan(
            minibatch_step, init=(rng, actor, critic), xs=mb_indices)
        return (rng, actor, critic), mb_metrics

    (rng, actor, critic), all_metrics = jax.lax.scan(
        epoch_step, init=(rng, actor, critic), length=num_epochs)

    metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
    metrics.update({
        "misc/reward_mean": rollout.rewards.mean(),
        "misc/advantages_mean": flat_adv.mean(),
        "misc/advantages_std": flat_adv.std(axis=0).mean(),
    })
    return rng, actor, critic, metrics


class DPPOAgent(BaseAgent):
    """
    Diffusion Policy Policy Optimization (DPPO)
    https://arxiv.org/abs/2409.00588
    """
    name = "DPPOAgent"
    model_names = ["actor", "critic"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DPPOConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        backbone_cls = {"mlp": MLP, "simba": Simba}[cfg.backbone_cls]
        actor_activation = get_activation(cfg.diffusion.activation)
        critic_activation = get_activation(cfg.critic_activation)

        backbone_def = ContinuousDDPMBackbone(
            noise_predictor=backbone_cls(
                hidden_dims=cfg.diffusion.hidden_dims, 
                output_dim=act_dim,
                activation=actor_activation),
            time_embedding=LearnableFourierEmbedding(
                output_dim=cfg.diffusion.time_dim),
            cond_embedding=MLP(
                hidden_dims=(128, 128), 
                activation=actor_activation),
        )
        self.actor = ContinuousDDPM.create(
            network=backbone_def, 
            rng=actor_rng,
            inputs=(jnp.ones((1, act_dim)), jnp.zeros((1, 1)), jnp.ones((1, obs_dim))),
            x_dim=act_dim, 
            steps=cfg.diffusion.steps,
            noise_schedule=cfg.diffusion.noise_schedule, 
            noise_schedule_params={},
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min, 
            x_max=cfg.diffusion.x_max,
            t_schedule_n=1.0,
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

        critic_def = ScalarCritic(
            backbone=backbone_cls(
            hidden_dims=cfg.critic_hidden_dims, 
            activation=critic_activation))
        self.critic = Model.create(
            critic_def, 
            critic_rng, 
            inputs=(jnp.ones((1, obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

    def train_step(self, rollout: RolloutBatch, step: int) -> Metric:
        self.rng, self.actor, self.critic, metrics = jit_update_dppo(
            self.rng, 
            self.actor, 
            self.critic, 
            rollout,
            gamma=self.cfg.gamma, 
            gae_lambda=self.cfg.gae_lambda,
            gamma_denoising=self.cfg.gamma_denoising,
            clip_epsilon=self.cfg.clip_epsilon,
            clip_epsilon_base=self.cfg.clip_epsilon_base,
            clip_epsilon_rate=self.cfg.clip_epsilon_rate,
            reward_scaling=self.cfg.reward_scaling,
            normalize_advantage=self.cfg.normalize_advantage,
            num_epochs=self.cfg.num_epochs,
            num_minibatches=self.cfg.num_minibatches,
            batch_size=self.cfg.batch_size,
            denoising_steps=self.cfg.diffusion.steps,
            min_logprob_std=self.cfg.diffusion.min_logprob_denoising_std,
        )
        return metrics

    def sample_actions(
        self, obs: jnp.ndarray, deterministic: bool = True, num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        assert num_samples == 1, "DPPO only supports num_samples=1"
        self.rng, action, chain, log_prob = jit_sample_actions(
            self.rng, 
            self.actor, 
            obs,
            self.cfg.diffusion.steps, 
            self.cfg.diffusion.min_logprob_denoising_std,
        )
        return action, {"log_prob": log_prob, "action_chains": chain}
