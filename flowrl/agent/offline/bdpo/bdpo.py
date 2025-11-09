from functools import partial
from typing import List
import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import scipy

from flowrl.agent.base import BaseAgent
from flowrl.config.offline.d4rl.algo.bdpo import (
    BDPOConfig,
    BDPODiffusionConfig,
    BDPODiffusionTrainConfig,
)
from flowrl.flow.ddpm import DDPM, DDPMBackbone, jit_update_ddpm
from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.critic import EnsembleCritic, EnsembleCriticT
from flowrl.module.mlp import MLP, ResidualMLP
from flowrl.module.model import Model
from flowrl.module.time_embedding import PositionalEmbedding
from flowrl.types import *
from flowrl.utils.plot_toy2d import energy_sample, SAMPLE_GRAPH_SIZE
from flowrl.dataset.toy2d import Toy2dDataset

EPS = 1e-6

@partial(jax.jit, static_argnames=("maxQ", "q_target", "rho"))
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

@partial(jax.jit, static_argnames=("T"))
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
) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
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

@partial(jax.jit, static_argnames=("T", "discount", "eta", "rho", "num_q_samples", "q_target", "maxQ", "solver", "ema", "do_ema_update"))
def jit_update_critic(
    rng: PRNGKey,
    q0: Model,
    q0_target: Model,
    vt: Model,
    vt_target: Model,
    actor_target: DDPM,
    behavior_target: DDPM,
    batch: Batch,
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
    rng, next_action, history = actor_target.sample(
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
    q0_target_value = batch.reward + discount * (1-batch.terminal) * q0_target_value

    q0_xt, q0_actor_eps = history
    q0_t = jnp.arange(T, 0, -1).repeat(B*num_q_samples, axis=0).reshape(T, B, num_q_samples, 1)
    q0_behavior_eps = behavior_target(
        q0_xt,
        q0_t,
        next_obs_repeat[jnp.newaxis, ...].repeat(T, axis=0),
    )
    q0_penalty = get_penalty(q0_actor_eps, q0_behavior_eps, q0_t, T, alphas, alpha_hats, betas)
    q0_penalty = q0_penalty.sum(axis=0).mean(axis=-2).sum(axis=-1, keepdims=True)

    q0_target_value = q0_target_value - eta * q0_penalty

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
    ) # (E, B, N, 1)
    rng, vt_target_value1 = get_target(rng, vt_target_value1, False, q_target, rho)
    vt_target_value2 = q0_target(
        obs_repeat,
        rep_xt_1
    )
    rng, vt_target_value2 = get_target(rng, vt_target_value2, False, q_target, rho)
    vt_target_value = (t != 1) * vt_target_value1 + (t == 1) * vt_target_value2

    vt_actor_eps = history
    vt_behavior_eps = behavior_target(xt, t, batch.obs)
    vt_penalty = get_penalty(vt_actor_eps, vt_behavior_eps, t, T, alphas, alpha_hats, betas)

    vt_target_value = vt_target_value - eta * vt_penalty.sum(axis=-1, keepdims=True)

    def q0_loss_fn(q0_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        pred = q0.apply(
            {"params": q0_params},
            batch.obs,
            batch.action
        )
        loss = ((pred - q0_target_value[jnp.newaxis, :])**2).mean()
        return loss, {
            "loss/q0_loss": loss,
            "misc/q0_mean": pred.mean(),
            "misc/q0_penalty": q0_penalty.mean(),
            "misc/reward": batch.reward.mean(),
        }
    def vt_loss_fn(vt_params: Param, *args, **kwargs) -> Tuple[jnp.ndarray, Metric]:
        pred = vt.apply(
            {"params": vt_params},
            batch.obs,
            xt,
            t
        )
        loss = ((pred - vt_target_value[jnp.newaxis, :])**2).mean()
        return loss, {
            "loss/vt_loss": loss,
            "misc/vt_mean": pred.mean(),
            "misc/vt_std": pred.std(axis=0).mean(),
            "misc/vt_penalty": vt_penalty.mean(),
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

@partial(jax.jit, static_argnames=("T", "eta", "rho", "num_q_samples", "q_target", "maxQ", "solver", "ema", "do_ema_update"))
def jit_update_actor(
    rng: PRNGKey,
    actor: DDPM,
    actor_target: DDPM,
    behavior_target: DDPM,
    q0_target: Model,
    vt_target: Model,
    batch: Batch,
    T: int,
    eta,
    rho: float,
    num_q_samples: int,
    q_target: str,
    maxQ: bool,
    solver: str,
    ema: float,
    do_ema_update: bool,
) -> Tuple[PRNGKey, Model, Model, Metric]:
    B = batch.obs.shape[0]
    A = batch.action.shape[-1]
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
        )
        sample_rng_, target_1 = get_target(sample_rng_, target_1, False, q_target, rho=rho)
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


class BDPOAgent(BaseAgent):
    """
    Behavior-Regularized Diffusion Policy Optimization
    """
    name = "BDPOAgent"
    model_names = ["behavior", "behavior_target", "actor", "actor_target", "q0", "q0_target", "vt", "vt_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: BDPOConfig, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg
        self.rng, behavior_rng, actor_rng, q0_rng, vt_rng = jax.random.split(self.rng, 5)

        # define behavior and actor
        time_embedding = partial(PositionalEmbedding, output_dim=cfg.diffusion.time_dim)
        if cfg.diffusion.resnet:
            noise_predictor = partial(
                ResidualMLP,
                hidden_dims=cfg.diffusion.resnet_hidden_dims,
                output_dim=act_dim,
                activation=mish,
                layer_norm=True,
                dropout=cfg.diffusion.dropout,
                multiplier=1,
            )
        else:
            noise_predictor = partial(
                MLP,
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
        q0_def = EnsembleCritic(
            hidden_dims=cfg.critic.hidden_dims,
            activation=mish,
            ensemble_size=cfg.critic.ensemble_size,
            layer_norm=cfg.critic.layer_norm,
        )
        vt_def = EnsembleCriticT(
            time_embedding=time_embedding,
            hidden_dims=cfg.critic.hidden_dims,
            activation=mish,
            ensemble_size=cfg.critic.ensemble_size,
            layer_norm=True,
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
        metrics = {}
        self.rng, self.q0, self.q0_target, self.vt, self.vt_target, critic_metrics = jit_update_critic(
            self.rng,
            self.q0,
            self.q0_target,
            self.vt,
            self.vt_target,
            self.actor_target,
            self.behavior_target,
            batch,
            self.cfg.diffusion.steps,
            self.cfg.critic.discount,
            self.cfg.critic.eta,
            self.cfg.critic.rho,
            self.cfg.critic.num_samples,
            self.cfg.critic.q_target,
            self.cfg.critic.maxQ,
            self.cfg.critic.solver,
            self.cfg.critic.ema,
            self._n_training_steps % self.cfg.critic.ema_every == 0
        )
        metrics.update(critic_metrics)

        if self._n_training_steps >= self.warmup_steps\
            and self._n_training_steps % self.cfg.critic.update_ratio == 0:
            self.rng, self.actor, self.actor_target, actor_metrics = jit_update_actor(
                self.rng,
                self.actor,
                self.actor_target,
                self.behavior_target,
                self.q0_target,
                self.vt_target,
                batch,
                self.cfg.diffusion.steps,
                self.cfg.critic.eta,
                self.cfg.critic.rho,
                self.cfg.critic.num_samples,
                self.cfg.critic.q_target,
                self.cfg.critic.maxQ,
                self.cfg.critic.solver,
                self.cfg.diffusion.actor.ema,
                self._n_training_steps % self.cfg.diffusion.actor.ema_every == 0
            )
            self.rng, self.behavior, self.behavior_target, behavior_metrics = jit_update_behavior(
                self.rng,
                self.behavior,
                self.behavior_target,
                batch,
                self.cfg.diffusion.behavior.ema,
                self._n_pretraining_steps % self.cfg.diffusion.behavior.ema_every == 0
            )
            metrics.update(actor_metrics)
            metrics.update(behavior_metrics)
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

    def plot_toy2d(self, save_dir: str, task: str) -> Metric:
        """
        Plot BDPO-specific visualizations for toy2d tasks.
        This includes value functions at different timesteps and value function error metrics.
        """
        metrics = {}
        os.makedirs(save_dir, exist_ok=True)
        self._plot_value(save_dir, task)
        metrics.update(self._test_value(task, from_dataset=True))
        metrics.update(self._test_value(task, from_dataset=False))
        return metrics

    def _plot_value(self, out_dir: str, task: str):
        """Plot value function over the action space at different timesteps."""
        plt.rc('font', family='Times New Roman', size=16)
        
        x_range = np.linspace(-4.5, 4.5, 90)
        y_range = np.linspace(-4.5, 4.5, 90)
        a, b = np.meshgrid(x_range, y_range, indexing='ij')
        id_matrix = np.stack([b, a], axis=-1).reshape(-1, 2)
        zero = np.zeros((90*90, 1))
        
        # Get vmin and vmax for color scale
        data, e = energy_sample(task, 0, sample_per_state=SAMPLE_GRAPH_SIZE)
        vmin = e.min()
        vmax = e.max()
        
        # BDPO agent has q0 (value at t=0) and vt (value at t=T)
        tt = [0, 1, 3, 5, 10, 20, 30, 40, 50]
        
        plt.figure(figsize=(42, 3.0))
        axes = []
        
        for i, t in enumerate(tt):
            plt.subplot(1, len(tt), i+1)
            
            if t == 0:
                # Use q0 (value function at t=0)
                critic_model = self.q0_target
                c = critic_model(zero, id_matrix).mean(0).reshape(90, 90)
            else:
                # Use vt (critic at different timesteps)
                critic_model = self.vt_target
                t_input = np.ones((90*90, 1)) * t
                c = critic_model(zero, id_matrix, t_input).mean(0).reshape(90, 90)
            
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(0, 89)
            plt.ylim(0, 89)
            
            if i == 0:
                mappable = plt.imshow(
                    c, origin='lower', vmin=vmin, vmax=vmax, 
                    cmap="winter", rasterized=True
                )
                plt.yticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])
            else:
                plt.imshow(
                    c, origin='lower', vmin=vmin, vmax=vmax, 
                    cmap="winter", rasterized=True
                )
                plt.yticks(ticks=[5, 25, 45, 65, 85], labels=[None, None, None, None, None])
            
            axes.append(plt.gca())
            plt.xticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])
            plt.title(f't={t}')
            
        plt.tight_layout()
        cbar = plt.gcf().colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
        import matplotlib
        plt.gcf().axes[-1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        saveto = os.path.join(out_dir, "qt_space.png")
        plt.savefig(saveto, dpi=300)
        plt.close()
        tqdm.write(f"Saved value plot to {saveto}")

    def _test_value(self, task: str, from_dataset: bool) -> tuple:
        """
        Compute the logmeanexp of value of dataset or actor generated samples,
        compare it to the target value.
        """
        datanum = 10000
        
        if from_dataset:
            # Get values from dataset
            dataset = Toy2dDataset(task=task, data_size=datanum, scan=False)
            batch = dataset.sample(batch_size=datanum)
            values = batch.reward.flatten()
        else:
            # Get values from actor samples
            samples, _ = self.sample_actions(
                obs=np.zeros((datanum, 1)),
                deterministic=False,
                num_samples=10,
            )
            
            # Compute value using the appropriate value function
            values = self.q0_target(
                np.zeros((datanum, 1)),
                samples,
            ).mean(0)
        eta = self.cfg.critic.eta
        T = self.cfg.critic.steps

        # Compute target as logsumexp
        target = jax.scipy.special.logsumexp(
            values / (eta / T), 
            b=(1 / datanum)
        ) * (eta / T)
        
        # Compute value at T using random samples
        N = 1000
        random_actions = jax.random.normal(jax.random.PRNGKey(0), shape=(N, 2))
        value_at_T = self.vt_target(
            np.zeros((N, 1)),
            random_actions,
            np.ones((N, 1)) * T,
        ).mean()
        abs_error = np.abs(target - value_at_T)
        rel_error = abs_error / (np.abs(target) + 1e-8)
        
        prefix = "dataset" if from_dataset else "actor"
        return {
            f"{prefix}_abs_error": float(abs_error),
            f"{prefix}_rel_error": float(rel_error),
            f"{prefix}_value_at_T": float(value_at_T),
            f"{prefix}_target": float(target)
        }
