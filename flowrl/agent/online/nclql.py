from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.online.algo.nclql import NCLQLConfig
from flowrl.flow.langevin_dynamics import AnnealedLangevinDynamics
from flowrl.functional.activation import mish
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
    # critic: Model,
    q1: Model,
    q2: Model,
    obs: jnp.ndarray,
    num_samples: int,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]

    def model_fn(x, l, condition=None):
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        cond_flat = condition.reshape(-1, condition.shape[-1]) if condition is not None else None
        t_flat = jnp.full((x_flat.shape[0], 1), l)

        def q_fn_single(action, cond, t_single):
            q1_pred = q1(cond[None], action[None], t_single[None])
            q2_pred = q2(cond[None], action[None], t_single[None])
            return ((q1_pred + q2_pred) / 2).squeeze()
            # qs = critic(cond[None], action[None], t_single[None])  # (ensemble, 1, 1)
            # return qs.mean(axis=0).squeeze()  # scalar

        energy_flat, q_grad_flat = jax.vmap(jax.value_and_grad(q_fn_single))(
            x_flat, cond_flat, t_flat
        )
        energy = energy_flat.reshape(batch_shape)
        q_grad = q_grad_flat.reshape(x.shape)
        return energy, q_grad

    if num_samples == 1:
        rng, x_init_rng = jax.random.split(rng)
        x_init = jax.random.uniform(x_init_rng, (B, ald.x_dim), minval=ald.x_min, maxval=ald.x_max)
        rng, actions, history = ald.sample(rng, model_fn, x_init, condition=obs)
    else:
        obs_repeat = jnp.repeat(obs, num_samples, axis=0)  # (B*num_samples, obs_dim)
        rng, x_init_rng = jax.random.split(rng)
        x_init = jax.random.uniform(x_init_rng, (B * num_samples, ald.x_dim), minval=ald.x_min, maxval=ald.x_max)
        rng, actions_flat, history = ald.sample(rng, model_fn, x_init, condition=obs_repeat)

        # select best action per obs using min-Q at final noise level
        l_final = jnp.full((B * num_samples, 1), ald.levels - 1)
        qs = critic(obs_repeat, actions_flat, l_final)  # (ensemble, B*num_samples, 1)
        qs_min = qs.min(axis=0).squeeze(-1).reshape(B, num_samples)
        best_idx = qs_min.argmax(axis=-1)
        actions = actions_flat.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]

    return rng, actions, history

@partial(jax.jit, static_argnames=("reward_scale", "num_samples", "discount", "ema"))
def jit_update_nclql(
    rng: PRNGKey,
    ald: AnnealedLangevinDynamics,
    q1: Model,
    q1_target: Model,
    q2: Model,
    q2_target: Model,
    batch: Batch,
    reward_scale: float,
    num_samples: int,
    discount: float,
    ema: float,
) -> Tuple[PRNGKey, Model, Model, Model, Metric]:

    batch.reward = batch.reward * reward_scale
    batch_size = batch.obs.shape[0]

    rng, next_action, _ = jit_sample_actions(
        rng,
        ald,
        q1,
        q2,
        batch.next_obs,
        num_samples,
    )

    # v_target = value_target(batch.next_obs, next_action)
    # v_target = batch.reward + discount * (1 - batch.terminal) * v_target.min(axis=0)
    # def value_loss_fn(value_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
    #     v_pred = value.apply({"params": value_params}, batch.obs, batch.action, training=True, rngs={"dropout": dropout_rng})
    #     value_loss = ((v_pred - v_target[jnp.newaxis, :])**2).mean()
    #     return value_loss, {
    #         "loss/value_loss": value_loss,
    #         "misc/value_mean": v_pred.mean(),
    #     }
    # value, value_metrics = value.apply_gradient(value_loss_fn)
    # value_target = ema_update(value, value_target, ema)

    # rng, l_rng, noise_rng = jax.random.split(rng, 3)
    # l = jax.random.randint(l_rng, (batch_size, 1), 0, ald.levels)
    # noise = jax.random.normal(noise_rng, (batch_size, ald.x_dim))
    # sigmas_l = ald.sigmas[l]
    # al = batch.action + sigmas_l * noise
    # l_target = value_target(batch.obs, batch.action).mean(axis=0)

    l_final = jnp.full((batch_size, 1), ald.levels - 1)
    td_target1 = q1_target(batch.next_obs, next_action, l_final)
    td_target2 = q2_target(batch.next_obs, next_action, l_final)
    td_target = jnp.minimum(td_target1, td_target2)
    td_target = batch.reward + discount * (1 - batch.terminal) * td_target

    rng, l_rng, noise_rng = jax.random.split(rng, 3)
    l = jax.random.randint(l_rng, (batch_size, 1), 0, ald.levels-1)
    noise = jax.random.normal(noise_rng, (batch_size, ald.x_dim))
    sigmas_l = ald.sigmas[l]
    al = batch.action + sigmas_l * noise
    l_target1 = q1_target(batch.obs, batch.action, l_final)
    l_target2 = q2_target(batch.obs, batch.action, l_final)
    l_target = (l_target1 + l_target2) / 2

    def td_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        td_pred = q1.apply(
            {"params": critic_params},
            batch.obs,
            batch.action,
            l_final,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        td_loss = ((td_pred - td_target)**2).mean()
        return td_loss, {
            "loss/td_loss": td_loss,
            "misc/td_pred": td_pred.mean(),
            "misc/td_target": td_target.mean(),
        }

    q1, td_metrics = q1.apply_gradient(td_loss_fn)
    q2, td_metrics = q2.apply_gradient(td_loss_fn)

    def perturb_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        l_pred = q1.apply(
            {"params": critic_params},
            batch.obs,
            al,
            l,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        perturb_loss = ((l_pred - l_target)**2).mean()
        return perturb_loss, {
            "loss/perturb_loss": perturb_loss,
            "misc/ql_pred": l_pred.mean(),
        }

    q1, perturb_metrics = q1.apply_gradient(perturb_loss_fn)
    q2, perturb_metrics = q2.apply_gradient(perturb_loss_fn)
    q1_target = ema_update(q1, q1_target, ema)
    q2_target = ema_update(q2, q2_target, ema)
    metrics = {
        # **value_metrics,
        **td_metrics,
        **perturb_metrics,
    }
    return rng, q1, q1_target, q2, q2_target, metrics


@partial(jax.jit, static_argnames=("num_samples",))
def jit_compute_statistics(
    rng: PRNGKey,
    ald: AnnealedLangevinDynamics,
    # critic: Model,
    q1: Model,
    q2: Model,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    num_samples: int = 256,
) -> Tuple[PRNGKey, Metric]:
    """Compute Q gradient L1 norms across noise levels using uniform random actions."""
    B = obs.shape[0]
    rng, action_rng = jax.random.split(rng)
    # actions = jax.random.uniform(action_rng, (num_samples, ald.x_dim), minval=-1.0, maxval=1.0)

    # broadcast: (B, num_samples, obs_dim), (B, num_samples, act_dim)
    # obs_expand = jnp.broadcast_to(obs[:, None, :], (B, num_samples, obs.shape[-1]))
    # actions_expand = jnp.broadcast_to(actions[None, :, :], (B, num_samples, ald.x_dim))

    # obs_flat = obs_expand.reshape(-1, obs.shape[-1])
    # actions_flat = actions_expand.reshape(-1, ald.x_dim)
    obs_flat = obs.reshape(-1, obs.shape[-1])
    actions_flat = actions.reshape(-1, ald.x_dim)

    def q_fn_single(action, cond, t):
        q1_pred = q1(cond[None], action[None], t[None])
        q2_pred = q2(cond[None], action[None], t[None])
        return ((q1_pred + q2_pred) / 2).squeeze()
        # qs = critic(cond[None], action[None], t[None])  # (ensemble, 1, 1)
        # return qs.mean(axis=0).squeeze()

    def grad_norm_at_level(l):
        t_flat = jnp.full((obs_flat.shape[0], 1), l)
        _, q_grad = jax.vmap(jax.value_and_grad(q_fn_single))(actions_flat, obs_flat, t_flat)
        return jnp.abs(q_grad).mean()

    metrics = {}
    for l in range(ald.levels):
        metrics[f"grad_detail/q_grad_l1_level_{l}"] = grad_norm_at_level(l)
    return rng, metrics


class NCLQLAgent(BaseAgent):
    """
    Noise-Conditioned Learning Q-Learning (NCLQL) agent.
    """
    name = "NCLQLAgent"
    model_names = ["critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: NCLQLConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, critic_rng, ald_rng = jax.random.split(self.rng, 3)

        # define the critic
        critic_activation = {
            "relu": jax.nn.relu,
            "elu": jax.nn.elu,
            "mish": mish,
        }[cfg.critic_activation]
        critic1_rng, critic2_rng = jax.random.split(critic_rng, 2)
        critic_def = ScalarCriticWithDiscreteTime(
            backbone=MLP(
                hidden_dims=cfg.critic_hidden_dims,
                activation=critic_activation,
                layer_norm=False,
                dropout=None,
            ),
            time_embedding=partial(PositionalEmbedding, output_dim=cfg.time_dim),
        )
        self.q1 = Model.create(
            critic_def,
            critic1_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.q1_target = Model.create(
            critic_def,
            critic1_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
        )
        self.q2 = Model.create(
            critic_def,
            critic2_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
        )
        self.q2_target = Model.create(
            critic_def,
            critic2_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.ones((1, 1))),
        )
        # critic_def = Ensemblize(
        #     base=BasicCriticWithDiscreteTime(
        #         backbone=MLP(
        #             hidden_dims=cfg.critic_hidden_dims,
        #             activation=critic_activation,
        #             layer_norm=False,
        #             dropout=None,
        #         ),
        #         time_embedding=partial(
        #             PositionalEmbedding,
        #             output_dim=cfg.time_dim
        #         ),
        #     ),
        #     ensemble_size=cfg.critic_ensemble_size,
        # )
        # self.critic = Model.create(
        #     critic_def,
        #     critic_rng,
        #     inputs=(
        #         jnp.ones((1, self.obs_dim)),
        #         jnp.ones((1, self.act_dim)),
        #         jnp.ones((1, 1)),
        #     ),
        #     optimizer=optax.adam(learning_rate=cfg.critic_lr),
        # )
        # self.critic_target = Model.create(
        #     critic_def,
        #     critic_rng,
        #     inputs=(
        #         jnp.ones((1, self.obs_dim)),
        #         jnp.ones((1, self.act_dim)),
        #         jnp.ones((1, 1)),
        #     ),
        # )

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
            x_max=cfg.ald.x_max
        )

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.q1, self.q1_target, self.q2, self.q2_target, metrics = jit_update_nclql(
            self.rng,
            self.ald,
            self.q1,
            self.q1_target,
            self.q2,
            self.q2_target,
            batch,
            reward_scale=self.cfg.reward_scale,
            num_samples=self.cfg.num_samples,
            discount=self.cfg.discount,
            ema=self.cfg.ema,
        )
        if self._n_training_steps % 1000 == 0:
            self.rng, stat_metrics = jit_compute_statistics(
                self.rng, self.ald, self.q1, self.q2, batch.obs, batch.action,
            )
            metrics.update(stat_metrics)
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
        self.rng, action, _ = jit_sample_actions(
            self.rng,
            self.ald,
            self.q1,
            self.q2,
            obs,
            num_samples=1
        )
        return action, {}
