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
    "discount", "ema", "num_train_samples",
    "noise_dim", "x_min", "x_max",
))
def jit_update_driftpo(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    critic_target: Model,
    batch: Batch,
    h_t: jnp.ndarray,
    lam: jnp.ndarray,
    base_step_size: jnp.ndarray,
    discount: float,
    ema: float,
    num_train_samples: int,
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
    # obs repeated once; shared by both the positive and negative MC batches
    obs_rep = batch.obs[jnp.newaxis, ...].repeat(num_train_samples, axis=0)
    # a^- ~ pi (current policy) for the kernelized mean-shift V^-
    neg_eps = jax.random.normal(neg_eps_rng, (num_train_samples, B, noise_dim))
    a_neg = jnp.clip(actor(jnp.concatenate([obs_rep, neg_eps], axis=-1)), x_min, x_max)
    a_neg = jax.lax.stop_gradient(a_neg)
    eps = jax.random.normal(eps_rng, (B, noise_dim))
    # neighborhood perturbations a^+ = a + sqrt(h_t) * z for V^+
    pos_noise = jax.random.normal(pos_noise_rng, (num_train_samples, B, act_dim))

    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        a_raw = actor.apply(
            {"params": actor_params},
            jnp.concatenate([batch.obs, eps], axis=-1),
            training=True,
            rngs={"dropout": dropout_rng},
        )
        a = clamp_ste(a_raw, x_min, x_max)
        a_sg = jax.lax.stop_gradient(a)

        def q_reduce(a_in: jnp.ndarray) -> jnp.ndarray:
            # ensemble-mean Q(s, a_in); shape (num_train_samples, B, 1)
            return new_critic(obs_rep, a_in).mean(axis=0)

        # Attraction toward the Boltzmann target pi* propto exp(Q/lam): a
        # softmax(Q/lam)-weighted average of grad_a Q over a neighborhood of a
        # (a "denoised" critic gradient). Sampling a^+ ~ N(a, h_t I) lets the
        # importance weights reduce to softmax(Q/lam). [score g in the notes]
        a_pos = jnp.clip(a_sg[jnp.newaxis, ...] + jnp.sqrt(h_t) * pos_noise, x_min, x_max)
        boltzmann_weight = jax.nn.softmax(q_reduce(a_pos) / lam, axis=0)  # (N, B, 1)
        grad_q = jax.grad(lambda a_in: q_reduce(a_in).sum())(a_pos)
        attraction = (boltzmann_weight * grad_q).sum(axis=0)  # (B, act_dim)

        # Repulsion away from the current policy: a kernel-weighted mean-shift
        # toward this state's own action samples a^- ~ pi, which we *subtract* to
        # spread actions apart and prevent collapse. [mean-shift m in the notes]
        to_neg = a_neg - a_sg[jnp.newaxis, ...]
        kernel = jnp.exp(-(to_neg ** 2).sum(axis=-1, keepdims=True) / (2.0 * h_t))
        repulsion = (kernel * to_neg).sum(axis=0) / (kernel.sum(axis=0) + 1e-8)  # (B, act_dim)

        # Drift = attraction - repulsion. Both are divided by the same scale
        # (mean magnitude of the attraction) so the direction still matches the
        # true score difference s^+ - s^- -- the fixed point pi = pi* is kept --
        # while the step becomes invariant to the critic-gradient magnitude. lam
        # (fixed) sets the attraction/repulsion balance; the effective step size
        # is base_step_size * h_t, shrinking as the bandwidth h_t anneals.
        scale = jnp.abs(attraction).mean() + 1e-8
        V = base_step_size * h_t * (attraction / scale - (lam / h_t / scale) * repulsion)
        V = jax.lax.stop_gradient(V)

        target = jnp.clip(a_sg + V, x_min, x_max)
        loss = ((a - target) ** 2).mean()
        return loss, {
            "loss/actor_loss": loss,
            "misc/attraction_norm": jnp.linalg.norm(attraction, axis=-1).mean(),
            "misc/repulsion_norm": jnp.linalg.norm(repulsion, axis=-1).mean(),
            "misc/drift_scale": scale,
            "misc/V_norm": jnp.linalg.norm(V, axis=-1).mean(),
            "misc/h_t": h_t,
            "misc/lambda": lam,
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
    sg(a + V(a; s)). The drift V combines an attraction toward the Boltzmann
    target pi* propto exp(Q/lam) and a repulsion away from the current policy:

        attraction = sum_j softmax(Q_j/lam) grad_a Q_j      (a^+ ~ N(a, h_t I))
        repulsion  = sum_l k(a, a^-_l)(a^-_l - a) / sum_l k  (a^- ~ pi)
        scale      = mean|attraction|
        V = base_step_size * h_t * (attraction / scale - (lam / h_t / scale) * repulsion).

    Dividing both terms by the same scale keeps V proportional to the true score
    difference s^+ - s^- = (1/lam) attraction - (1/h_t) repulsion, so the fixed
    point pi = pi* is preserved for every h_t > 0, while the step is invariant to
    the critic-gradient magnitude. lam (fixed, the only per-env knob) sets the
    attraction / repulsion balance; h_t is annealed and also scales the effective
    step base_step_size * h_t.
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

    def _bandwidth(self) -> float:
        # linear anneal bandwidth_init -> bandwidth_end over bandwidth_anneal_steps
        frac = min(self._n_training_steps / max(self.cfg.bandwidth_anneal_steps, 1), 1.0)
        return self.cfg.bandwidth_init + (self.cfg.bandwidth_end - self.cfg.bandwidth_init) * frac

    def train_step(self, batch: Batch, step: int) -> Metric:
        h_t = jnp.asarray(self._bandwidth(), dtype=jnp.float32)
        lam = jnp.asarray(self.cfg.temp, dtype=jnp.float32)
        base_step_size = jnp.asarray(self.cfg.base_step_size, dtype=jnp.float32)
        self.rng, self.actor, self.critic, self.critic_target, metrics = jit_update_driftpo(
            self.rng,
            self.actor,
            self.critic,
            self.critic_target,
            batch,
            h_t,
            lam,
            base_step_size,
            discount=self.cfg.discount,
            ema=self.cfg.ema,
            num_train_samples=self.cfg.num_train_samples,
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
