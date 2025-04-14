from functools import partial
from typing import Tuple
import jax
from flowrl.config.d4rl.algo.iql import IQLConfig
from flowrl.functional.ema import ema_udpate
from flowrl.functional.loss import expectile_regression
from flowrl.module.mlp import MLP
from flowrl.module.actor import SquashedDeterministicActor
from flowrl.module.critic import EnsembleCritic
from flowrl.types import Batch, Metric, PRNGKey, Param
from flowrl.module.model import Model
from .base import BaseAgent
import jax.numpy as jnp
import optax

@jax.jit
def _jit_sample_action(actor: Model, obs: jnp.ndarray) -> jnp.ndarray:
    """
    Sample action from the actor network.
    """
    actor, action = actor(obs)
    return actor, action

def _update_v(
    q: jnp.ndarray,
    value: Model,
    batch: Batch,
    expectile: float,
) -> Tuple[Model, Metric]:
    def value_loss_fn(value_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        v = value.apply(
            {'params': value_params},
            batch.obs,
            training=True,
            rngs={'dropout': dropout_rng},
        )
        value_loss = expectile_regression(v, q, expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_std': v.std(),
            'v_norm': jnp.linalg.norm(v, axis=-1).mean(),
        }
    return value.apply_gradient(value_loss_fn)

def awr_update_actor(
    actor: Model,
    q: jnp.array,
    v: jnp.array,
    batch: Batch,
    beta: float
) -> Tuple[Model, Metric]:
    exp_a = jnp.exp((q - v) * beta)
    exp_a = jnp.minimum(exp_a, 100.0)  # truncate the weights...
    def actor_loss_fn(actor_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        # it creates a function of params, so it has to directly use net.apply() function rather than
        # the __call__() function of the Model object "actor"
        # here training is True for this forward pass is used for gradient descent
        pred = actor.apply(
            {'params': actor_params},
            batch.obs,
            training=True,
            rngs={'dropout': dropout_rng},
        )
        actor_loss = jnp.mean((pred - batch.action) ** 2)
        adv = q - v
        return actor_loss, {
            'actor_loss': actor_loss, 
            'adv_mean': adv.mean(),
            'adv_std': adv.std(),
            'adv_norm': jnp.linalg.norm(adv, axis=-1).mean(),
        }
    return actor.apply_gradient(actor_loss_fn)

def update_q(
    critic: Model,
    next_v: jnp.array,
    batch: Batch,
    discount: float
) -> Tuple[Model, Metric]:
    target_q = batch.reward + discount * (1-batch.terminal) * next_v
    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q1, q2 = critic.apply(
            {'params': critic_params},
            batch.obs,
            batch.action,
            training=True,
            rngs={'dropout': dropout_rng},
        )
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1_mean': q1.mean(),
            'q1_std': q1.std(),
            'q1_norm': jnp.linalg.norm(q1, axis=-1).mean(),
            'q2_mean': q2.mean(),
            'q2_std': q2.std(),
            'q2_norm': jnp.linalg.norm(q2, axis=-1).mean(),
        }
    return critic.apply_gradient(critic_loss_fn)

@partial(jax.jit, static_argnames=("expectile", "discount", "tau"))
def _jit_update(
    actor: Model,
    critic: Model,
    critic_target: Model,
    value: Model,
    batch: Batch,
    expectile: float,
    discount: float,
    tau: float
)-> Tuple[Model, Model, Model, Model, Metric]:
    critic_target, (q1, q2) = critic_target(batch.obs, batch.action)
    q = jnp.minimum(q1, q2)
    new_value, value_metric = _update_v(q, value, batch, expectile)
    new_value, v = new_value(batch.obs)
    new_value, next_v = new_value(batch.next_obs)
    new_actor, actor_metric = awr_update_actor(
        actor, q, v, batch, expectile
    )
    new_critic, critic_metric = update_q(
        critic, next_v, batch, discount
    )
    new_target_critic = ema_udpate(new_critic, critic_target, tau)
    return new_actor, new_critic, new_target_critic, new_value, {
        **actor_metric,
        **critic_metric,
        **value_metric,
    }

class IQLAgent(BaseAgent):
    """
    Implicit Q-Learning (IQL) agent.
    """
    name = "IQLAgent"
    model_names = []

    def __init__(self, obs_dim: int, act_dim: int, cfg: IQLConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, actor_key, critic_key, value_key = jax.random.split(self.rng, 4)

        actor_def = SquashedDeterministicActor(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.layer_norm,
                dropout=cfg.dropout
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim
        )
        if cfg.opt_decay_schedule == "cosine" and cfg.lr_decay_steps is not None:
            schedule_fn = optax.cosine_decay_schedule(-cfg.actor_lr, cfg.lr_decay_steps)
            act_opt = optax.chain(optax.scale_by_adam(),
                                  optax.scale_by_schedule(schedule_fn))
        else:
            act_opt = optax.adam(learning_rate=cfg.actor_lr)
        self.actor = Model.create(
            actor_def,
            actor_key,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=act_opt,
            clip_grad_norm=cfg.clip_grad_norm,
        )

        critic_def = EnsembleCritic(
            hidden_dims=cfg.critic_hidden_dims,
            layer_norm=cfg.layer_norm,
            dropout=cfg.dropout,
            ensemble_size=cfg.critic_ensemble_size
        )
        self.critic = Model.create(
            critic_def,
            critic_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic_target = Model.create(
            critic_def,
            critic_key,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )

        value_def = EnsembleCritic(
            hidden_dims=cfg.value_hidden_dims,
            layer_norm=cfg.layer_norm,
            dropout=cfg.dropout,
        )
        self.value = Model.create(
            value_def,
            value_key,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.value_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )

    def train_step(self, batch, step: int):
        self.actor, self.critic, self.critic_target, self.value, metric = _jit_update(
            self.actor,
            self.critic,
            self.critic_target,
            self.value,
            batch,
            self.cfg.expectile,
            self.cfg.discount,
            self.cfg.tau,
        )
        return metric
    
    def sample_actions(self, obs, use_behavior = False, temperature = 0, num_samples = 1, return_history = False):
        assert not use_behavior, "IQL have no behavior policy"
        assert num_samples==1, "IQL only supports num_samples=1"
        assert not return_history, "IQL does not support return_history"
        self.actor, action = _jit_sample_action(self.actor, obs)
        return action, None
