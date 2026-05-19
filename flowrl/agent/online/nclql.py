from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.nclql import NCLQLConfig
from flowrl.flow.langevin_dynamics import AnnealedLangevinDynamics
from flowrl.functional.activation import get_activation
from flowrl.functional.ema import ema_update
from flowrl.module.critic import Ensemblize, ScalarCriticWithDiscreteTime
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("num_samples",))
def jit_sample_actions(
    rng: PRNGKey,
    ald: AnnealedLangevinDynamics,
    critic: Model,
    obs: jnp.ndarray,
    num_samples: int,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, x_init_rng = jax.random.split(rng)

    if num_samples == 1:
        cond = obs
        x_init = jax.random.uniform(x_init_rng, (B, ald.x_dim), minval=ald.x_min, maxval=ald.x_max)
    else:
        cond = jnp.broadcast_to(obs[:, jnp.newaxis, :], (B, num_samples, obs.shape[-1]))
        x_init = jax.random.uniform(x_init_rng, (B, num_samples, ald.x_dim), minval=ald.x_min, maxval=ald.x_max)

    def model_fn(x, l, condition=None):
        t = jnp.full((*x.shape[:-1], 1), l)
        def q_sum(action):
            return critic(condition, action, t).sum()
        q_grad = jax.grad(q_sum)(x)
        energy = critic(condition, x, t).mean(axis=0).squeeze(-1)
        return energy, q_grad

    rng, actions, _ = ald.sample(rng, model_fn, x_init, condition=cond)

    if num_samples > 1:
        l_final = jnp.full((B, num_samples, 1), ald.levels - 1)
        qs = critic(cond, actions, l_final).min(axis=0).squeeze(-1)
        best_idx = qs.argmax(axis=-1)
        actions = actions[jnp.arange(B), best_idx]
    return rng, actions

@partial(jax.jit, static_argnames=("reward_scale", "discount", "ema"))
def jit_update_nclql(
    rng: PRNGKey,
    ald: AnnealedLangevinDynamics,
    critic: Model,
    critic_target: Model,
    batch: Batch,
    reward_scale: float,
    discount: float,
    ema: float,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    rng, sample_rng, l_rng, noise_rng = jax.random.split(rng, 4)
    B = batch.obs.shape[0]
    reward = batch.reward * reward_scale

    # TD target at the terminal noise level using current critic for sampling and target critic for value
    _, next_action = jit_sample_actions(sample_rng, ald, critic, batch.next_obs, num_samples=1)
    l_final = jnp.full((B, 1), ald.levels - 1)
    q_next = critic_target(batch.next_obs, next_action, l_final).min(axis=0)
    td_target = reward + discount * (1 - batch.terminal) * q_next

    # denoising target = mean ensemble Q at level L-1, snapshot before any update
    q_mean_target = critic(batch.obs, batch.action, l_final).mean(axis=0)

    def td_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q = critic.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            l_final,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        td_loss = ((q - td_target[jnp.newaxis, :]) ** 2).mean()
        return td_loss, {
            "loss/td_loss": td_loss,
            "misc/q_mean": q.mean(),
            "misc/td_target": td_target.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, td_metrics = critic.apply_gradient(td_loss_fn)

    # perturb action and train q at random level l < L-1 to denoise back to q at L-1
    l = jax.random.randint(l_rng, (B, 1), 0, ald.levels - 1)
    noise = jax.random.normal(noise_rng, (B, ald.x_dim))
    sigmas_l = ald.sigmas[l]
    al = batch.action + sigmas_l * noise

    def perturb_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q_t = new_critic.apply(
            {"params": critic_params},
            batch.obs,
            al,
            l,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        perturb_loss = ((q_t - q_mean_target[jnp.newaxis, :]) ** 2).mean()
        return perturb_loss, {
            "loss/perturb_loss": perturb_loss,
            "misc/q_t_mean": q_t.mean(),
        }

    new_critic, perturb_metrics = new_critic.apply_gradient(perturb_loss_fn)
    new_critic_target = ema_update(new_critic, critic_target, ema)
    return rng, new_critic, new_critic_target, {
        **td_metrics,
        **perturb_metrics,
    }


class NCLQLAgent(BaseAgent):
    """
    Noise-Conditioned Langevin Q-Learning (NCLQL).
    """
    name = "NCLQLAgent"
    model_names = ["critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: NCLQLConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, critic_rng, ald_rng = jax.random.split(self.rng, 3)

        critic_activation = get_activation(cfg.critic_activation)

        # define the critic
        critic_def = Ensemblize(
            base=ScalarCriticWithDiscreteTime(
                backbone=MLP(
                    hidden_dims=cfg.critic_hidden_dims,
                    activation=critic_activation,
                ),
                time_embedding=PositionalEmbedding(output_dim=cfg.time_dim),
            ),
            ensemble_size=cfg.critic_ensemble_size,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
        )

        self.ald = AnnealedLangevinDynamics.create(
            rng=ald_rng,
            x_dim=self.act_dim,
            steps=cfg.ald.steps,
            levels=cfg.ald.levels,
            w=cfg.ald.w,
            sigma_max=cfg.ald.sigma_max,
            sigma_min=cfg.ald.sigma_min,
            step_lr=cfg.ald.step_lr,
            q_grad_norm=cfg.ald.q_grad_norm,
            clip_sampler=cfg.ald.clip_sampler,
            x_min=cfg.ald.x_min,
            x_max=cfg.ald.x_max,
        )

        # define tracking variables
        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.critic, self.critic_target, metrics = jit_update_nclql(
            self.rng,
            self.ald,
            self.critic,
            self.critic_target,
            batch,
            reward_scale=self.cfg.reward_scale,
            discount=self.cfg.discount,
            ema=self.cfg.ema,
        )
        self._n_training_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        # if deterministic is true, sample cfg.num_samples actions and select the best one
        # if not, sample 1 action
        if deterministic:
            num_samples = self.cfg.num_samples
        else:
            num_samples = 1
        self.rng, action = jit_sample_actions(
            self.rng,
            self.ald,
            self.critic,
            obs,
            num_samples=num_samples,
        )
        return action, {}
