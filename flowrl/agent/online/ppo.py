from functools import partial

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.ppo import PPOConfig
from flowrl.module.actor import GaussianActor
from flowrl.module.critic import ScalarCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.simba import Simba
from flowrl.types import Metric, PRNGKey, RolloutBatch

# --------------------------------------------------------------------------- #
# Reusable PPO building blocks
# --------------------------------------------------------------------------- #

LR_MIN = 1e-5
LR_MAX = 1e-2


def compute_gae(
    terminated: jnp.ndarray,   # (T, B, 1)
    truncated: jnp.ndarray,    # (T, B, 1)
    rewards: jnp.ndarray,      # (T, B, 1)
    values: jnp.ndarray,       # (T, B, 1)
    next_values: jnp.ndarray,  # (T, B, 1)
    gae_lambda: float,
    gamma: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """rsl_rl-style GAE with V(s_t) time-out bootstrap.

    For truncated steps we approximate V(s_{t+1}) ≈ V(s_t) by folding
    gamma * V(s_t) into the reward and then treating the step as done
    (so the usual next_values term drops out).
    """
    rewards = rewards + gamma * values * truncated
    dones = jnp.maximum(terminated, truncated)
    next_is_not_terminal = 1.0 - dones
    deltas = rewards + gamma * next_is_not_terminal * next_values - values
    gae_discount = next_is_not_terminal * gamma * gae_lambda

    def gae_step(carry, x):
        delta_t, disc_t = x
        a_t = delta_t + disc_t * carry
        return a_t, a_t

    _, advantages = jax.lax.scan(
        gae_step,
        init=jnp.zeros_like(values[0]),
        xs=(deltas[::-1], gae_discount[::-1]),
    )
    advantages = advantages[::-1]
    lambda_returns = advantages + values
    return lambda_returns, advantages


def normalize_advantages(advantages: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def ppo_surrogate_loss(
    new_log_probs: jnp.ndarray,  # (N, 1)
    old_log_probs: jnp.ndarray,  # (N, 1)
    advantages: jnp.ndarray,     # (N, 1)
    clip_epsilon: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Standard PPO clipped surrogate. Returns (loss, ratio, clipped_fraction)."""
    ratio = jnp.exp(new_log_probs - old_log_probs)
    surrogate_unclipped = ratio * advantages
    surrogate_clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -jnp.minimum(surrogate_unclipped, surrogate_clipped).mean()
    clipped_fraction = (jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32).mean()
    return policy_loss, ratio, clipped_fraction


def clipped_value_loss(
    v_new: jnp.ndarray,
    v_old: jnp.ndarray,
    returns: jnp.ndarray,
    clip_epsilon: float,
    use_clipped: bool,
) -> jnp.ndarray:
    """rsl_rl-style clipped critic loss: max((v-R)^2, (v_clipped-R)^2)."""
    if use_clipped:
        v_clipped = v_old + jnp.clip(v_new - v_old, -clip_epsilon, clip_epsilon)
        loss_unclipped = (v_new - returns) ** 2
        loss_clipped_ = (v_clipped - returns) ** 2
        return jnp.maximum(loss_unclipped, loss_clipped_).mean()
    return ((v_new - returns) ** 2).mean()


def gaussian_kl_diag(
    mu0: jnp.ndarray,
    std0: jnp.ndarray,
    mu1: jnp.ndarray,
    std1: jnp.ndarray,
) -> jnp.ndarray:
    """KL( N(mu0, diag(std0^2)) || N(mu1, diag(std1^2)) ) summed over the last dim."""
    return jnp.sum(
        jnp.log(std1 / std0)
        + (std0 ** 2 + (mu0 - mu1) ** 2) / (2.0 * std1 ** 2)
        - 0.5,
        axis=-1,
    )


def _set_lr_in_opt_state(opt_state, new_lr):
    """Set learning_rate in a chain(clip_by_global_norm, inject_hyperparams(opt)) state.

    Model.create wraps the base optimizer with ``optax.clip_by_global_norm`` when
    ``clip_grad_norm`` is truthy, so ``opt_state`` is a 2-tuple whose second entry
    is an ``optax.InjectHyperparamsState``.
    """
    inject = opt_state[1]
    new_hp = dict(inject.hyperparams)
    new_hp["learning_rate"] = new_lr
    return (opt_state[0], inject._replace(hyperparams=new_hp))


def apply_lr_to_model(model: Model, new_lr: jnp.ndarray) -> Model:
    new_opt_state = _set_lr_in_opt_state(model.state.opt_state, new_lr)
    return model.replace(state=model.state.replace(opt_state=new_opt_state))


def adaptive_lr_update(
    kl_mean: jnp.ndarray,
    actor_lr: jnp.ndarray,
    critic_lr: jnp.ndarray,
    desired_kl: float,
    lr_min: float = LR_MIN,
    lr_max: float = LR_MAX,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """rsl_rl-style adaptive KL schedule. Scales both LRs by the same factor."""
    factor = jnp.where(
        kl_mean > desired_kl * 2.0,
        1.0 / 1.5,
        jnp.where(
            (kl_mean < desired_kl / 2.0) & (kl_mean > 0.0),
            1.5,
            1.0,
        ),
    )
    return (
        jnp.clip(actor_lr * factor, lr_min, lr_max),
        jnp.clip(critic_lr * factor, lr_min, lr_max),
    )


def make_minibatch_indices(
    rng: jax.Array,
    total: int,
    num_minibatches: int,
    batch_size: int,
) -> jnp.ndarray:
    """Shuffle indices and slice into (num_minibatches, batch_size)."""
    perm = jax.random.permutation(rng, total)
    return perm[: num_minibatches * batch_size].reshape(num_minibatches, batch_size)


# --------------------------------------------------------------------------- #
# Vanilla Gaussian PPO
# --------------------------------------------------------------------------- #


@partial(jax.jit, static_argnames=(
    "gamma", "gae_lambda", "clip_epsilon", "entropy_coeff", "value_loss_coef",
    "use_clipped_value_loss", "normalize_advantage", "adaptive", "desired_kl",
    "num_epochs", "num_minibatches", "batch_size",
))
def jit_update_ppo(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    rollout: RolloutBatch,
    actor_lr: jnp.ndarray,
    critic_lr: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    entropy_coeff: float,
    value_loss_coef: float,
    use_clipped_value_loss: bool,
    normalize_advantage: bool,
    adaptive: bool,
    desired_kl: float,
    num_epochs: int,
    num_minibatches: int,
    batch_size: int,
):
    T, B = rollout.rewards.shape[:2]

    # Values computed with rollout-time critic (no updates yet this step).
    old_values = critic(rollout.obs)            # (T, B, 1)
    next_old_values = critic(rollout.next_obs)  # (T, B, 1)

    lambda_returns, advantages = jax.lax.stop_gradient(
        compute_gae(
            terminated=rollout.terminated,
            truncated=rollout.truncated,
            rewards=rollout.rewards,
            values=old_values,
            next_values=next_old_values,
            gae_lambda=gae_lambda,
            gamma=gamma,
        )
    )
    if normalize_advantage:
        advantages = normalize_advantages(advantages)

    flat_obs = rollout.obs.reshape(T * B, -1)
    flat_actions = rollout.actions.reshape(T * B, -1)
    flat_old_log_probs = rollout.extras["log_prob"].reshape(T * B, 1)
    flat_old_means = rollout.extras["mean"].reshape(T * B, -1)
    flat_old_stds = rollout.extras["std"].reshape(T * B, -1)
    flat_advantages = advantages.reshape(T * B, 1)
    flat_returns = lambda_returns.reshape(T * B, 1)
    flat_old_values = old_values.reshape(T * B, 1)

    def epoch_step(carry, _):
        rng, actor, critic, actor_lr, critic_lr = carry
        rng, perm_rng = jax.random.split(rng)
        mb_indices = make_minibatch_indices(perm_rng, T * B, num_minibatches, batch_size)

        def minibatch_step(carry, indices):
            rng, actor, critic, actor_lr, critic_lr = carry

            mb_obs = flat_obs[indices]
            mb_actions = flat_actions[indices]
            mb_old_log_probs = flat_old_log_probs[indices]
            mb_old_means = flat_old_means[indices]
            mb_old_stds = flat_old_stds[indices]
            mb_advantages = flat_advantages[indices]
            mb_returns = flat_returns[indices]
            mb_old_values = flat_old_values[indices]

            # --- Adaptive LR schedule (per-minibatch) ---------------------------
            if adaptive:
                probe_dist = actor(mb_obs)
                kl = gaussian_kl_diag(
                    jax.lax.stop_gradient(mb_old_means),
                    jax.lax.stop_gradient(mb_old_stds),
                    jax.lax.stop_gradient(probe_dist.mean()),
                    jax.lax.stop_gradient(probe_dist.stddev()),
                )
                kl_mean = jnp.mean(kl)
                actor_lr, critic_lr = adaptive_lr_update(
                    kl_mean, actor_lr, critic_lr, desired_kl
                )
                actor = apply_lr_to_model(actor, actor_lr)
                critic = apply_lr_to_model(critic, critic_lr)
            else:
                kl_mean = jnp.zeros((), dtype=jnp.float32)

            # --- Actor loss -----------------------------------------------------
            def actor_loss_fn(actor_params, dropout_rng):
                dist = actor.apply(
                    {"params": actor_params},
                    mb_obs,
                    training=True,
                    rngs={"dropout": dropout_rng},
                )
                new_log_probs = dist.log_prob(mb_actions)[..., jnp.newaxis]
                policy_loss, ratio, clipped_frac = ppo_surrogate_loss(
                    new_log_probs, mb_old_log_probs, mb_advantages, clip_epsilon
                )
                entropy = dist.entropy().mean()
                entropy_loss = -entropy_coeff * entropy
                return policy_loss + entropy_loss, {
                    "loss/policy_loss": policy_loss,
                    "loss/entropy_loss": entropy_loss,
                    "misc/entropy": entropy,
                    "misc/policy_ratio": ratio.mean(),
                    "misc/clipped_ratio": clipped_frac,
                    "misc/action_mean": dist.mean().mean(),
                    "misc/action_l1": jnp.abs(dist.mean()).mean(),
                    "misc/action_std": dist.stddev().mean(),
                }

            new_actor, actor_metrics = actor.apply_gradient(actor_loss_fn)

            # --- Critic loss ----------------------------------------------------
            def critic_loss_fn(critic_params, dropout_rng):
                v = critic.apply(
                    {"params": critic_params},
                    mb_obs,
                    training=True,
                    rngs={"dropout": dropout_rng},
                )
                v_loss_raw = clipped_value_loss(
                    v, mb_old_values, mb_returns, clip_epsilon, use_clipped_value_loss
                )
                return value_loss_coef * v_loss_raw, {
                    "loss/value_loss": v_loss_raw,
                    "misc/value_mean": v.mean(),
                }

            new_critic, critic_metrics = critic.apply_gradient(critic_loss_fn)

            metrics = {
                **actor_metrics,
                **critic_metrics,
                "misc/kl_mean": kl_mean,
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
        "misc/advantages_mean": flat_advantages.mean(),
        "misc/advantages_std": flat_advantages.std(axis=0).mean(),
        "misc/returns_mean": flat_returns.mean(),
    })
    return rng, actor, critic, actor_lr, critic_lr, metrics


@partial(jax.jit, static_argnames=("deterministic",))
def jit_sample_action(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    obs: jnp.ndarray,
    deterministic: bool,
):
    dist = actor(obs, training=False)
    mean = dist.mean()
    std = jnp.broadcast_to(dist.stddev(), mean.shape)
    if deterministic:
        action = mean
        log_prob = jnp.zeros(obs.shape[0])
    else:
        action, log_prob = dist.sample_and_log_prob(seed=rng)
    value = critic(obs, training=False)
    return action, log_prob, mean, std, value


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO)
    """

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

        actor_def = GaussianActor(
            backbone=backbone_cls(
                hidden_dims=cfg.actor_hidden_dims,
                activation=activation,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
            conditional_logstd=False,
            logstd_min=-20.0,
            logstd_max=2.0,
        )
        critic_def = ScalarCritic(
            backbone=backbone_cls(
                hidden_dims=cfg.critic_hidden_dims,
                activation=activation,
            ),
        )

        # Use inject_hyperparams so the adaptive schedule can mutate LR inside jit.
        actor_opt = optax.inject_hyperparams(optax.adam)(learning_rate=cfg.actor_lr)
        critic_opt = optax.inject_hyperparams(optax.adam)(learning_rate=cfg.critic_lr)

        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=actor_opt,
            clip_grad_norm=cfg.max_grad_norm,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=critic_opt,
            clip_grad_norm=cfg.max_grad_norm,
        )

        # Default logstd init is zeros (=> std=1); override if init_noise_std != 1.
        if abs(cfg.init_noise_std - 1.0) > 1e-8:
            self.actor = self._override_init_noise_std(self.actor, cfg.init_noise_std)

        # Running LRs; mutated in-place by the adaptive schedule each train_step.
        self.actor_lr = jnp.asarray(cfg.actor_lr, dtype=jnp.float32)
        self.critic_lr = jnp.asarray(cfg.critic_lr, dtype=jnp.float32)

    @staticmethod
    def _override_init_noise_std(actor: Model, init_noise_std: float) -> Model:
        target = jnp.log(jnp.asarray(init_noise_std, dtype=jnp.float32))
        params = dict(actor.state.params)
        params["logstd"] = jnp.full_like(params["logstd"], target)
        new_state = actor.state.replace(params=params)
        return actor.replace(state=new_state)

    def train_step(self, rollout: RolloutBatch, step: int) -> Metric:
        batch_size = self.cfg.num_envs * self.cfg.rollout_length // self.cfg.num_minibatches
        (
            self.rng,
            self.actor,
            self.critic,
            self.actor_lr,
            self.critic_lr,
            metrics,
        ) = jit_update_ppo(
            self.rng,
            self.actor,
            self.critic,
            rollout,
            self.actor_lr,
            self.critic_lr,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_epsilon=self.cfg.clip_epsilon,
            entropy_coeff=self.cfg.entropy_coeff,
            value_loss_coef=self.cfg.value_loss_coef,
            use_clipped_value_loss=self.cfg.use_clipped_value_loss,
            normalize_advantage=self.cfg.normalize_advantage,
            adaptive=(self.cfg.schedule == "adaptive"),
            desired_kl=self.cfg.desired_kl,
            num_epochs=self.cfg.num_epochs,
            num_minibatches=self.cfg.num_minibatches,
            batch_size=batch_size,
        )
        return metrics

    def sample_actions(self, obs, deterministic=True, num_samples=1):
        assert num_samples == 1, "PPO only supports num_samples=1"
        self.rng, sample_key = jax.random.split(self.rng)
        action, log_prob, mean, std, value = jit_sample_action(
            sample_key,
            self.actor,
            self.critic,
            obs,
            deterministic,
        )
        return action, {
            "log_prob": log_prob[..., jnp.newaxis],
            "mean": mean,
            "std": std,
            "value": value,
        }
