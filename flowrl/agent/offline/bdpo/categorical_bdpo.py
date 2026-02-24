from functools import partial
from typing import List

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from flowrl.agent.base import BaseAgent
from flowrl.config.offline.algo.categorical_bdpo import (
    CategoricalBDPOConfig,
    BDPODiffusionConfig,
    BDPODiffusionTrainConfig,
)
from flowrl.flow.ddpm import DDPM, DDPMBackbone, jit_update_ddpm
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.functional.math import symlog, symexp
from flowrl.module.critic import CategoricalCriticWithDiscreteTime, Ensemblize, ScalarCritic
from flowrl.module.mlp import MLP, ResidualMLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import *

EPS = 1e-6

def get_target(rng: PRNGKey, vs: jnp.ndarray, maxQ: bool, q_target: str, rho: float) -> Tuple[PRNGKey, jnp.ndarray]:
    # vs is of shape (E, B, N, 1)
    if maxQ:
        vs = vs.max(axis=-2)
    else:
        vs = vs.mean(axis=-2)
    if q_target == "min":
        vs = vs.min(axis=0)
    elif q_target == "convex":
        vs = rho * vs.min(axis=0) + (1-rho) * vs.max(axis=0)
    elif q_target == "lcb":
        vs = vs.mean(axis=0) - rho * vs.std(axis=0)
    elif q_target == "rand_convex":
        rng, alpha_key = jax.random.split(rng)
        alphas = jax.random.uniform(alpha_key, vs.shape)
        alphas /= (alphas.sum(axis=0, keepdims=True) + EPS)
        vs = (vs * alphas).sum(axis=0)
    else:
        raise NotImplementedError(f"Unrecognized Q-target type: {q_target}. ")
    return rng, vs

def get_penalty(
    actor_eps,
    behavior_eps,
    t: jnp.ndarray,
    T: int,
    alphas: jnp.ndarray,
    alpha_hats: jnp.ndarray,
    betas: jnp.ndarray
) -> jnp.ndarray:
    return 0.5 * betas[t] * ((actor_eps - behavior_eps)**2) / (1 - betas[t]) / (1 - alpha_hats[t])

def get_atoms(v_min: float, v_max: float, num_atoms: int) -> jnp.ndarray:
    """Get support atoms (pre-transformation) for categorical distribution."""
    return jnp.linspace(v_min, v_max, num_atoms)

def logits_to_value(
    logits: jnp.ndarray,
    v_min: float,
    v_max: float,
    num_atoms: int
) -> jnp.ndarray:
    """Expected value (post-transformation) from categorical logits.

    Args:
        logits: raw logits, shape (E, B, N, num_atoms)
        v_min: minimum support value
        v_max: maximum support value
        num_atoms: number of atoms in the categorical distribution
    Returns:
        expected value, shape (B, 1)
    """
    # The computation proceeds in the following order:
    #   1. softmax over atoms
    #   2. mean over ensemble members
    #   3. compute expected value
    #   4. mean over num_q_samples dimension
    #   5. symexp
    # Steps 1, 2, and 3 are in the correct relative order, consistent with BRC.
    # Discussion or the order: https://github.com/typoverflow/flow-rl/pull/25#discussion_r2839940571
    E, B, N, _ = logits.shape # assertion for the shape of logits, will raise an error if the shape is not as expected
    atoms = get_atoms(v_min, v_max, num_atoms) # (num_atoms,)
    probs = jax.nn.softmax(logits, axis=-1).mean(axis=0) # (B, N, num_atoms)
    value =  (probs * atoms).sum(axis=-1, keepdims=True) # (B, N, 1)
    value = value.mean(axis=1) # (B, 1)
    value = symexp(value) # (B, 1)
    return value

def categorical_project(
    target_values: jnp.ndarray,
    v_min: float,
    v_max: float,
    num_atoms: int
) -> Tuple[jnp.ndarray, Metric]:
    """C51-style projection: scalar targets (original-space) → categorical distribution.

    Args:
        target_values: shape (*B, 1)
        v_min: minimum support value
        v_max: maximum support value
        num_atoms: number of atoms in the categorical distribution
    Returns:
        target distribution, shape (*B, num_atoms)
        metrics dict with pre-clip statistics (keys: pre_clip_max, pre_clip_min,
            pre_clip_mean, clip_ratio; all computed in the transformed/symlog space)
    """
    delta_z = (v_max - v_min) / (num_atoms - 1)

    symlog_values = symlog(target_values)
    clip_ratio = ((symlog_values < v_min) | (symlog_values > v_max)).mean()
    metrics = {
        "pre_clip_max": symlog_values.max(),
        "pre_clip_min": symlog_values.min(),
        "pre_clip_mean": symlog_values.mean(),
        "clip_ratio": clip_ratio,
    }

    clipped = jnp.clip(symlog_values, v_min, v_max)
    b = (clipped - v_min) / delta_z # (*B, 1)
    l = jnp.floor(b).astype(jnp.int32) # (*B, 1)
    u = jnp.ceil(b).astype(jnp.int32) # (*B, 1)

    d_u = (b - l.astype(jnp.float32)) # (*B, 1)
    d_l = 1.0 - d_u # (*B, 1)

    probs = (d_l * jax.nn.one_hot(l.squeeze(-1), num_atoms)
           + d_u * jax.nn.one_hot(u.squeeze(-1), num_atoms))
    return probs, metrics

@partial(jax.jit, static_argnames=("training", "num_samples", "solver", "temperature"))
def jit_sample_and_select(
    rng: PRNGKey,
    model: DDPM,
    q0: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    solver: str,
    temperature: float,
) -> Tuple[PRNGKey, jnp.ndarray]:
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], model.x_dim))
    rng, actions, _ = model.sample(rng, xT, obs_repeat, training, solver)
    if temperature is None:
        return rng, actions[:, 0]
    else:
        qs = q0(
            obs_repeat,
            actions,
        )
        qs = qs.mean(axis=0).reshape(B, num_samples)
        if temperature <= 0.0:
            idx = qs.argmax(axis=-1)
        else:
            rng, select_rng = jax.random.split(rng)
            idx = jax.random.categorical(select_rng, logits=qs/(1e-6+temperature), axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), idx]
        return rng, actions

@partial(jax.jit, static_argnames=("ema", "do_ema_update"))
def jit_update_behavior(
    rng: PRNGKey,
    behavior: DDPM,
    behavior_target: DDPM,
    batch: Batch,
    ema: float,
    do_ema_update: bool,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    rng, new_behavior, metrics = jit_update_ddpm(rng, behavior, batch.action, batch.obs)
    if do_ema_update:
        new_behavior_target = ema_update(new_behavior, behavior_target, ema)
    else:
        new_behavior_target = behavior_target
    return rng, new_behavior, new_behavior_target, metrics

def update_critic(
    rng: PRNGKey,
    q0: Model,
    q0_target: Model,
    vt: Model,
    vt_target: Model,
    actor_target: DDPM,
    behavior_target: DDPM,
    batch: Batch,
    v_min: float,
    v_max: float,
    num_atoms: int,
    T: int,
    discount: float,
    eta: float,
    rho: float,
    num_q_samples: int,
    q_target: str,
    maxQ: bool,
    solver: str,
    ema: float,
    do_ema_update: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Metric]:
    B = batch.obs.shape[0]
    A = batch.action.shape[-1]
    alphas = actor_target.alphas
    alpha_hats = actor_target.alpha_hats
    betas = actor_target.betas

    rng, xT_rng = jax.random.split(rng)

    # q0 target
    next_obs_repeat = batch.next_obs[..., jnp.newaxis, :].repeat(num_q_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*next_obs_repeat.shape[:-1], A))
    rng, next_action, _ = actor_target.sample(
        rng,
        xT,
        next_obs_repeat,
        training=False,
        solver=solver,
    )
    q0_target_value = q0_target(
        next_obs_repeat,
        next_action
    )
    rng, q0_target_value = get_target(rng, q0_target_value, maxQ, q_target, rho)

    q0_target_value = batch.reward + discount * (1-batch.terminal) * q0_target_value # (B, 1)

    # vt target
    obs_repeat = batch.obs[..., jnp.newaxis, :].repeat(num_q_samples, axis=-2)
    rng, rep_xt_1, xt, rep_t_1, t, history = actor_target.onestep_sample(
        rng,
        batch.action,
        batch.obs,
        training=False,
        num_samples=num_q_samples,
        solver=solver,
        sample_xt=True,
        t=None,
    )
    vt_target_value1 = vt_target(
        obs_repeat,
        rep_xt_1,
        rep_t_1
    ) # (E, B, N, num_atoms)
    vt_target_value1 = logits_to_value(vt_target_value1, v_min, v_max, num_atoms) # (B, 1)
    vt_target_value2 = q0_target(
        obs_repeat,
        rep_xt_1
    )
    rng, vt_target_value2 = get_target(rng, vt_target_value2, False, q_target, rho)
    vt_target_value = (t != 1) * vt_target_value1 + (t == 1) * vt_target_value2

    vt_actor_eps = history
    vt_behavior_eps = behavior_target(xt, t, batch.obs)
    vt_penalty = get_penalty(vt_actor_eps, vt_behavior_eps, t, T, alphas, alpha_hats, betas)

    vt_target_value = vt_target_value - eta * vt_penalty.sum(axis=-1, keepdims=True) # (B, 1)
    vt_target_probs, vt_proj_metrics = categorical_project(vt_target_value, v_min, v_max, num_atoms) # (B, num_atoms)

    def q0_loss_fn(q0_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        pred = q0.apply(
            {"params": q0_params},
            batch.obs,
            batch.action
        ) # (E, B, 1)
        loss = ((pred - q0_target_value[jnp.newaxis, :])**2).mean()
        return loss, {
            "loss/q0_loss": loss,
            "misc/q0_mean": pred.mean(),
            "misc/reward": batch.reward.mean(),
        }
    def vt_loss_fn(vt_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        pred = vt.apply(
            {"params": vt_params},
            batch.obs,
            xt,
            t
        ) # (E, B, num_atoms)
        # sum over atoms (cross-entropy loss) → mean over batch → mean over ensemble members → scalar
        log_probs = jax.nn.log_softmax(pred, axis=-1)
        loss = -(vt_target_probs[jnp.newaxis, :] * log_probs).sum(axis=-1).mean()
        E, B, _ = pred.shape
        vt_value = logits_to_value(pred.reshape(E, B, 1, num_atoms), v_min, v_max, num_atoms) # (B, 1)
        return loss, {
            "loss/vt_loss": loss,
            "misc/vt_mean": vt_value.mean(),
            "misc/vt_logits_std": jax.nn.softmax(pred, axis=-1).std(axis=0).mean(),
            "misc/vt_penalty": vt_penalty.mean(),
            "misc/vt_target_symlog_max": vt_proj_metrics["pre_clip_max"],
            "misc/vt_target_symlog_min": vt_proj_metrics["pre_clip_min"],
            "misc/vt_target_symlog_mean": vt_proj_metrics["pre_clip_mean"],
            "misc/vt_target_clip_ratio": vt_proj_metrics["clip_ratio"],
        }
    new_q0, q0_metrics = q0.apply_gradient(q0_loss_fn)
    new_vt, vt_metrics = vt.apply_gradient(vt_loss_fn)
    if do_ema_update:
        new_q0_target = ema_update(new_q0, q0_target, ema)
        new_vt_target = ema_update(new_vt, vt_target, ema)
    else:
        new_q0_target = q0_target
        new_vt_target = vt_target
    return rng, new_q0, new_q0_target, new_vt, new_vt_target, {
        **q0_metrics,
        **vt_metrics
    }

def update_actor(
    rng: PRNGKey,
    actor: DDPM,
    actor_target: DDPM,
    behavior_target: DDPM,
    q0_target: Model,
    vt_target: Model,
    batch: Batch,
    v_min: float,
    v_max: float,
    num_atoms: int,
    T: int,
    eta,
    rho: float,
    num_q_samples: int,
    q_target: str,
    solver: str,
    ema: float,
    do_ema_update: bool,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    alphas = actor.alphas
    alpha_hats = actor.alpha_hats
    betas = actor.betas

    rng, sample_rng = jax.random.split(rng)

    def actor_loss_fn(actor_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        sample_rng_, rep_xt_1, xt, rep_t_1, t, history = actor.onestep_sample(
            sample_rng,
            batch.action,
            batch.obs,
            training=True,
            num_samples=num_q_samples,
            solver=solver,
            sample_xt=True,
            t=None,
            params=actor_params
        )
        obs_repeat = batch.obs[..., jnp.newaxis, :].repeat(num_q_samples, axis=-2)
        target_1 = vt_target(
            obs_repeat,
            rep_xt_1,
            rep_t_1
        ) # (E, B, N, num_atoms)
        target_1 = logits_to_value(target_1, v_min, v_max, num_atoms) # (B, 1)
        target_2 = q0_target(
            obs_repeat,
            rep_xt_1
        )
        sample_rng_, target_2 = get_target(sample_rng_, target_2, False, q_target, rho=rho)
        target = (t != 1) * target_1 + (t == 1) * target_2

        actor_eps = history
        behavior_eps = behavior_target(xt, t, batch.obs)
        penalty = get_penalty(actor_eps, behavior_eps, t, T, alphas, alpha_hats, betas)
        target = target - eta * penalty.sum(axis=-1, keepdims=True)
        actor_loss = - target.mean()
        return actor_loss, {
            "loss/actor_loss": actor_loss
        }

    new_actor, metrics = actor.apply_gradient(actor_loss_fn)
    if do_ema_update:
        new_actor_target = ema_update(new_actor, actor_target, ema)
    else:
        new_actor_target = actor_target
    metrics["misc/eta"] = eta
    return rng, new_actor, new_actor_target, metrics

@partial(jax.jit, static_argnames=(
    "do_actor_update",
    "v_min",
    "v_max",
    "num_atoms",
    "diffusion_steps",
    "discount",
    "eta",
    "rho",
    "num_q_samples",
    "q_target",
    "maxQ",
    "solver",
    "critic_ema",
    "do_critic_ema_update",
    "actor_ema",
    "do_actor_ema_update",
))
def jit_train_step(
    rng: PRNGKey,
    # models
    q0: Model,
    q0_target: Model,
    vt: Model,
    vt_target: Model,
    actor: DDPM,
    actor_target: DDPM,
    behavior_target: DDPM,
    batch: Batch,
    do_actor_update: bool,
    # categorical parameters
    v_min: float,
    v_max: float,
    num_atoms: int,
    # other hyperparameters
    diffusion_steps: int,
    discount: float,
    eta: float,
    rho: float,
    num_q_samples: int,
    q_target: str,
    maxQ: bool,
    solver: str,
    # ema update
    critic_ema: float,
    do_critic_ema_update: bool,
    actor_ema: float,
    do_actor_ema_update: bool,
):
    metrics = {}
    rng, q0, q0_target, vt, vt_target, critic_metrics = update_critic(
        rng=rng,
        q0=q0,
        q0_target=q0_target,
        vt=vt,
        vt_target=vt_target,
        actor_target=actor_target,
        behavior_target=behavior_target,
        batch=batch,
        v_min=v_min,
        v_max=v_max,
        num_atoms=num_atoms,
        T=diffusion_steps,
        discount=discount,
        eta=eta,
        rho=rho,
        num_q_samples=num_q_samples,
        q_target=q_target,
        maxQ=maxQ,
        solver=solver,
        ema=critic_ema,
        do_ema_update=do_critic_ema_update,
    )
    metrics.update(critic_metrics)
    if do_actor_update:
        rng, actor, actor_target, actor_metrics = update_actor(
            rng=rng,
            actor=actor,
            actor_target=actor_target,
            behavior_target=behavior_target,
            q0_target=q0_target,
            vt_target=vt_target,
            batch=batch,
            v_min=v_min,
            v_max=v_max,
            num_atoms=num_atoms,
            T=diffusion_steps,
            eta=eta,
            rho=rho,
            num_q_samples=num_q_samples,
            q_target=q_target,
            solver=solver,
            ema=actor_ema,
            do_ema_update=do_actor_ema_update,
        )
        metrics.update(actor_metrics)
    return rng, q0, q0_target, vt, vt_target, actor, actor_target, metrics
    

class CategoricalBDPOAgent(BaseAgent):
    """
    Categorical version of Behavior-Regularized Diffusion Policy Optimization (BDPO)
    https://arxiv.org/abs/2502.04778
    """
    name = "CategoricalBDPOAgent"
    model_names = ["behavior", "behavior_target", "actor", "actor_target", "q0", "q0_target", "vt", "vt_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: CategoricalBDPOConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, behavior_rng, actor_rng, q0_rng, vt_rng = jax.random.split(self.rng, 5)

        # define behavior and actor
        time_embedding = PositionalEmbedding(output_dim=cfg.diffusion.time_dim)
        if cfg.diffusion.resnet:
            noise_predictor = ResidualMLP(
                hidden_dims=cfg.diffusion.resnet_hidden_dims,
                output_dim=act_dim,
                activation=mish,
                layer_norm=True,
                dropout=cfg.diffusion.dropout,
                multiplier=1,
            )
        else:
            noise_predictor = MLP(
                hidden_dims=cfg.diffusion.mlp_hidden_dims,
                output_dim=act_dim,
                activation=mish,
                layer_norm=cfg.diffusion.layer_norm,
                dropout=cfg.diffusion.dropout
            )
        backbone_def = DDPMBackbone(
            noise_predictor=noise_predictor,
            time_embedding=time_embedding
        )

        def create_ddpm(network_def: nn.Module, rng: PRNGKey, cfg: BDPODiffusionConfig, train_cfg: BDPODiffusionTrainConfig):
            if train_cfg.lr_decay_steps is not None:
                lr = optax.cosine_decay_schedule(train_cfg.lr, train_cfg.lr_decay_steps)
            else:
                lr = train_cfg.lr
            ddpm = DDPM.create(
                network=network_def,
                rng=rng,
                inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
                x_dim=self.act_dim,
                steps=cfg.steps,
                noise_schedule=cfg.noise_schedule,
                noise_schedule_params=None,
                approx_postvar=True,
                clip_sampler=cfg.clip_sampler,
                x_min=cfg.x_min,
                x_max=cfg.x_max,
                optimizer=optax.adam(learning_rate=lr),
                clip_grad_norm=train_cfg.clip_grad_norm,
            )
            ddpm_target = DDPM.create(
                network=network_def,
                rng=rng,
                x_dim=self.act_dim,
                inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
                steps=cfg.steps,
                noise_schedule=cfg.noise_schedule,
                noise_schedule_params=None,
                approx_postvar=True,
                clip_sampler=cfg.clip_sampler,
                x_min=cfg.x_min,
                x_max=cfg.x_max,
            )
            return ddpm, ddpm_target

        self.behavior, self.behavior_target = create_ddpm(backbone_def, behavior_rng, cfg.diffusion, cfg.diffusion.behavior)
        self.actor, self.actor_target = create_ddpm(backbone_def, actor_rng, cfg.diffusion, cfg.diffusion.actor)

        # define critic networks
        if cfg.critic.lr_decay_steps is not None:
            q0_lr = optax.cosine_decay_schedule(cfg.critic.lr, cfg.critic.lr_decay_steps)
            vt_lr = optax.cosine_decay_schedule(cfg.critic.lr, cfg.critic.lr_decay_steps)
        else:
            q0_lr = cfg.critic.lr
            vt_lr = cfg.critic.lr
        q0_def = Ensemblize(
            base=ScalarCritic(
                backbone=MLP(
                    hidden_dims=cfg.critic.hidden_dims,
                    activation=mish,
                    layer_norm=cfg.critic.layer_norm,
                )
            ),
            ensemble_size=cfg.critic.ensemble_size,
        )
        vt_def = Ensemblize(
            base=CategoricalCriticWithDiscreteTime(
                backbone=MLP(
                    hidden_dims=cfg.critic.hidden_dims,
                    activation=mish,
                    layer_norm=True,
                ),
                time_embedding=time_embedding,
                output_nodes=cfg.critic.output_nodes,
            ),
            ensemble_size=cfg.critic.ensemble_size,
        )
        self.q0 = Model.create(
            q0_def,
            q0_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
            optimizer=optax.adam(learning_rate=q0_lr),
            clip_grad_norm=cfg.critic.clip_grad_norm,
        )
        self.q0_target = Model.create(
            q0_def,
            q0_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim))),
        )
        self.vt = Model.create(
            vt_def,
            vt_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.zeros((1, 1))),
            optimizer=optax.adam(learning_rate=vt_lr),
            clip_grad_norm = cfg.critic.clip_grad_norm,
        )
        self.vt_target = Model.create(
            vt_def,
            vt_rng,
            inputs=(jnp.ones((1, self.obs_dim)), jnp.ones((1, self.act_dim)), jnp.zeros((1, 1))),
        )

        self.warmup_steps = cfg.warmup_steps
        self._is_pretraining = True # will switch to False after prepared for training
        self._n_pretraining_steps = 0
        self._n_training_steps = 0

    @property
    def saved_model_names(self) -> List[str]:
        if self._is_pretraining:
            return ["behavior_target"]
        else:
            return self.model_names

    def prepare_training(self):
        self.actor = ema_update(self.behavior_target, self.actor, 1.0)
        self.actor_target = ema_update(self.behavior_target, self.actor_target, 1.0)
        self._is_pretraining = False

    def train_step(self, batch: Batch, step: int) -> Metric:
        do_actor_update = (self._n_training_steps >= self.warmup_steps)\
            and (self._n_training_steps % self.cfg.critic.update_ratio == 0)
        do_critic_ema_update = self._n_training_steps % self.cfg.critic.ema_every == 0
        if do_actor_update:
            actor_step = self._n_training_steps // self.cfg.critic.update_ratio
            do_actor_ema_update = actor_step % self.cfg.diffusion.actor.ema_every == 0
        else:
            do_actor_ema_update = False
        self.rng, self.q0, self.q0_target, self.vt, self.vt_target, self.actor, self.actor_target, metrics = jit_train_step(
            rng=self.rng,
            q0=self.q0,
            q0_target=self.q0_target,
            vt=self.vt,
            vt_target=self.vt_target,
            actor=self.actor,
            actor_target=self.actor_target,
            behavior_target=self.behavior_target,
            batch=batch,
            do_actor_update=do_actor_update,
            v_min=self.cfg.critic.v_min,
            v_max=self.cfg.critic.v_max,
            num_atoms=self.cfg.critic.output_nodes,
            diffusion_steps=self.cfg.diffusion.steps,
            discount=self.cfg.critic.discount,
            eta=self.cfg.critic.eta,
            rho=self.cfg.critic.rho,
            num_q_samples=self.cfg.critic.num_samples,
            q_target=self.cfg.critic.q_target,
            maxQ=self.cfg.critic.maxQ,
            solver=self.cfg.critic.solver,
            critic_ema=self.cfg.critic.ema,
            do_critic_ema_update=do_critic_ema_update,
            actor_ema=self.cfg.diffusion.actor.ema,
            do_actor_ema_update=do_actor_ema_update,
        )
        self._n_training_steps += 1
        return metrics

    def pretrain_step(self, batch: Batch, step: int) -> Metric:
        self.rng, self.behavior, self.behavior_target, metrics = jit_update_behavior(
            self.rng,
            self.behavior,
            self.behavior_target,
            batch,
            self.cfg.diffusion.behavior.ema,
            self._n_pretraining_steps % self.cfg.diffusion.behavior.ema_every == 0
        )
        self._n_pretraining_steps += 1
        return metrics

    def sample_actions(
        self,
        obs: jnp.ndarray,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> Tuple[jnp.ndarray, Metric]:
        if self._is_pretraining:
            use_model = self.behavior_target
            temperature = None
            num_samples = 1
        else:
            use_model = self.actor_target
            temperature = self.cfg.temperature
            num_samples = num_samples
        self.rng, action = jit_sample_and_select(
            self.rng,
            use_model,
            self.q0_target,
            obs,
            training=False,
            num_samples=num_samples,
            solver=self.cfg.diffusion.solver,
            temperature=temperature,
        )
        return action, {}
