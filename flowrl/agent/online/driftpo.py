from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.driftpo import DriftPOConfig
from flowrl.functional.activation import get_activation
from flowrl.functional.ema import ema_update
from flowrl.module.critic import Ensemblize, ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.types import Batch, Metric, Param, PRNGKey


def clamp_ste(x: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
    clamped = jnp.clip(x, min_val, max_val)
    return x + jax.lax.stop_gradient(clamped - x)


@partial(jax.jit, static_argnames=("noise_dim", "num_samples", "x_min", "x_max"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    obs: jnp.ndarray,
    noise_dim: int,
    num_samples: int,
    x_min: float,
    x_max: float,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, eps_rng = jax.random.split(rng)
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    eps = jax.random.normal(eps_rng, (*obs_repeat.shape[:-1], noise_dim))
    actions = actor(jnp.concatenate([obs_repeat, eps], axis=-1))
    actions = jnp.clip(actions, x_min, x_max)
    if num_samples == 1:
        actions = actions[:, 0]
    else:
        qs = critic(obs_repeat, actions).min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions[jnp.arange(B), best_idx]
    return rng, actions


@partial(jax.jit, static_argnames=(
    "discount", "ema",
    "opt_method", "num_pos_samples", "num_neg_samples",
    "bandwidth", "bandwidth_mode",
    "temp", "temp_mode", "pos_strength",
    "drift_scale", "max_step",
    "noise_dim", "x_min", "x_max",
))
def jit_update_driftpo(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    batch: Batch,
    discount: float,
    ema: float,
    opt_method: str,
    num_pos_samples: int,
    num_neg_samples: int,
    bandwidth: float,
    bandwidth_mode: str,
    temp: float,
    temp_mode: str,
    pos_strength: float,
    drift_scale: float,
    max_step: float,
    noise_dim: int,
    x_min: float,
    x_max: float,
) -> Tuple[PRNGKey, Model, Model, Model, Metric]:
    B = batch.obs.shape[0]
    act_dim = batch.action.shape[-1]

    rng, next_eps_rng, redq_rng = jax.random.split(rng, 3)
    next_eps = jax.random.normal(next_eps_rng, (B, noise_dim))
    next_action = actor(jnp.concatenate([batch.next_obs, next_eps], axis=-1))
    next_action = jnp.clip(next_action, x_min, x_max)
    q_next = critic_target(batch.next_obs, next_action)
    # TD target: REDQ-style min over a random 2-subset of the ensemble
    subset_idx = jax.random.choice(redq_rng, q_next.shape[0], shape=(2,), replace=False)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_next[subset_idx].min(axis=0)

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = ((q - q_target[jnp.newaxis, :]) ** 2).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

    rng, neg_eps_rng, eps_rng, pos_noise_rng = jax.random.split(rng, 4)
    # a^- ~ pi (current policy) for the kernelized mean-shift V^-
    neg_eps = jax.random.normal(neg_eps_rng, (num_neg_samples, B, noise_dim))
    obs_neg = batch.obs[jnp.newaxis, ...].repeat(num_neg_samples, axis=0)
    a_neg = jnp.clip(actor(jnp.concatenate([obs_neg, neg_eps], axis=-1)), x_min, x_max)
    a_neg = jax.lax.stop_gradient(a_neg)
    eps = jax.random.normal(eps_rng, (B, noise_dim))
    # perturbations a^+ = a + sqrt(h) * z for zeroth-order V^+
    pos_noise = jax.random.normal(pos_noise_rng, (num_pos_samples, B, act_dim))

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        a_raw = actor.apply(
            {"params": actor_params},
            jnp.concatenate([batch.obs, eps], axis=-1),
            training=True,
            rngs={"dropout": dropout_rng},
        )
        a = clamp_ste(a_raw, x_min, x_max)
        a_sg = jax.lax.stop_gradient(a)

        diff_neg = a_neg - a_sg[jnp.newaxis, ...]
        sq_dist_neg = (diff_neg ** 2).sum(axis=-1, keepdims=True)
        if bandwidth_mode == "median":
            med = jnp.median(sq_dist_neg, axis=0)
            denom = jnp.log(num_neg_samples + 1.0) * 2.0
            h_eff = jnp.maximum(med / denom, bandwidth)
        else:
            h_eff = jnp.full((a_sg.shape[0], 1), bandwidth)

        # V^-: kernelized mean-shift over current policy samples
        k_vals = jnp.exp(-sq_dist_neg / (2.0 * h_eff[jnp.newaxis, ...]))
        Z_neg = k_vals.sum(axis=0)
        V_neg = (k_vals * diff_neg).sum(axis=0) / (Z_neg + 1e-8)
        V_neg_norm = jnp.linalg.norm(V_neg, axis=-1, keepdims=True)

        # V^+: positive force toward pi* prop exp(Q / temp)
        if opt_method == "first":
            def q_sum(a_in):
                return new_critic(batch.obs, a_in).mean(axis=0).sum()
            grad_q = jax.grad(q_sum)(a_sg)
            if temp_mode == "normalize":
                lam = 1.0 / (jnp.abs(grad_q).mean() + 1e-8)
                V_pos = h_eff * lam * grad_q
            elif temp_mode == "balance":
                grad_q_norm = jnp.linalg.norm(grad_q, axis=-1, keepdims=True)
                V_pos = pos_strength * V_neg_norm * grad_q / (grad_q_norm + 1e-8)
            elif temp_mode == "raw":
                V_pos = grad_q
            else:
                V_pos = (h_eff / temp) * grad_q
        else:
            a_pos = a_sg[jnp.newaxis, ...] + jnp.sqrt(h_eff)[jnp.newaxis, ...] * pos_noise
            a_pos = jnp.clip(a_pos, x_min, x_max)
            obs_pos = batch.obs[jnp.newaxis, ...].repeat(num_pos_samples, axis=0)
            q_pos = new_critic(obs_pos, a_pos).min(axis=0)
            if temp_mode == "normalize":
                q_centered = (q_pos - q_pos.mean(axis=0, keepdims=True)) / (q_pos.std(axis=0, keepdims=True) + 1e-6)
                weights = jax.nn.softmax(q_centered / temp, axis=0)
                V_pos = (weights * (a_pos - a_sg[jnp.newaxis, ...])).sum(axis=0)
            elif temp_mode == "balance":
                q_centered = (q_pos - q_pos.mean(axis=0, keepdims=True)) / (q_pos.std(axis=0, keepdims=True) + 1e-6)
                weights = jax.nn.softmax(q_centered / temp, axis=0)
                V_pos_raw = (weights * (a_pos - a_sg[jnp.newaxis, ...])).sum(axis=0)
                V_pos_norm = jnp.linalg.norm(V_pos_raw, axis=-1, keepdims=True)
                V_pos = pos_strength * V_neg_norm * V_pos_raw / (V_pos_norm + 1e-8)
            else:
                weights = jax.nn.softmax(q_pos / temp, axis=0)
                V_pos = (weights * (a_pos - a_sg[jnp.newaxis, ...])).sum(axis=0)

        V_raw = drift_scale * (V_pos - V_neg)

        if max_step > 0.0:
            V_norm = jnp.linalg.norm(V_raw, axis=-1, keepdims=True)
            scale_factor = jnp.minimum(1.0, max_step / (V_norm + 1e-8))
            V = V_raw * scale_factor
            clipped_frac = (V_norm > max_step).astype(jnp.float32).mean()
        else:
            V = V_raw
            clipped_frac = jnp.array(0.0)

        V = jax.lax.stop_gradient(V)

        target = jnp.clip(a_sg + V, x_min, x_max)
        loss = ((a - target) ** 2).mean()
        V_pos_norm_metric = jnp.linalg.norm(V_pos, axis=-1).mean()
        V_neg_norm_metric = V_neg_norm.mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/V_pos_norm": V_pos_norm_metric,
            "misc/V_neg_norm": V_neg_norm_metric,
            "misc/V_pos_over_neg": V_pos_norm_metric / (V_neg_norm_metric + 1e-8),
            "misc/V_norm": jnp.linalg.norm(V, axis=-1).mean(),
            "misc/V_clipped_frac": clipped_frac,
            "misc/h_eff": h_eff.mean(),
            "misc/a_l1": jnp.abs(a).mean(),
        }

    new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)
    new_critic_target = ema_update(new_critic, critic_target, ema)

    return rng, new_actor, new_critic, new_critic_target, {
        **critic_metrics,
        **actor_metrics,
    }


class DriftPOAgent(BaseAgent):
    """
    Drifting Policy (DriftPO).

    One-step generative policy a = f(eps; s) trained by regressing toward
    a + V(a; s), where V = V^+ - V^- attracts samples toward pi* prop exp(Q/temp)
    and repels them from the current policy distribution (kernelized mean-shift).

    V^+ variants:
      - "first":  V^+ propto grad_a Q(s, a)  (Stein / W2-flow)
      - "zeroth": sample a^+ ~ k(.,a), reweight by exp(Q / temp)
    """
    name = "DriftPOAgent"
    model_names = ["actor", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DriftPOConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)

        backbone_cls = {
            "mlp": MLP,
            "simba": Simba,
        }[cfg.backbone_cls]
        actor_activation = get_activation(cfg.actor_activation)
        critic_activation = get_activation(cfg.critic_activation)

        actor_def = backbone_cls(
            hidden_dims=cfg.actor_hidden_dims,
            output_dim=act_dim,
            activation=actor_activation,
        )
        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, obs_dim + cfg.noise_dim)),),
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
        )

        critic_def = Ensemblize(
            base=ScalarCritic(
                backbone=backbone_cls(
                    hidden_dims=cfg.critic_hidden_dims,
                    activation=critic_activation,
                ),
            ),
            ensemble_size=cfg.critic_ensemble_size,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, obs_dim)), jnp.ones((1, act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, obs_dim)), jnp.ones((1, act_dim))),
        )

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.actor, self.critic, self.critic_target, metrics = jit_update_driftpo(
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            batch,
            discount=self.cfg.discount,
            ema=self.cfg.ema,
            opt_method=self.cfg.opt_method,
            num_pos_samples=self.cfg.num_pos_samples,
            num_neg_samples=self.cfg.num_neg_samples,
            bandwidth=self.cfg.bandwidth,
            bandwidth_mode=self.cfg.bandwidth_mode,
            temp=self.cfg.temp,
            temp_mode=self.cfg.temp_mode,
            pos_strength=self.cfg.pos_strength,
            drift_scale=self.cfg.drift_scale,
            max_step=self.cfg.max_step,
            noise_dim=self.cfg.noise_dim,
            x_min=self.cfg.x_min,
            x_max=self.cfg.x_max,
        )
        self._n_training_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        if deterministic:
            num_samples = self.cfg.num_samples
        else:
            num_samples = 1
        self.rng, action = jit_sample_actions(
            self.rng,
            self.actor,
            self.critic,
            obs,
            noise_dim=self.cfg.noise_dim,
            num_samples=num_samples,
            x_min=self.cfg.x_min,
            x_max=self.cfg.x_max,
        )
        return action, {}
