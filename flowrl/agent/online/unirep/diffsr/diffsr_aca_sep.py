from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from flowrl.agent.online.td3 import TD3Agent
from flowrl.agent.online.unirep.diffsr.network import (
    FactorizedDDPM,
    update_factorized_ddpm,
)
from flowrl.config.online.mujoco.algo.diffsr import DiffSRTD3Config
from flowrl.functional.activation import l2_normalize
from flowrl.functional.ema import ema_update
from flowrl.module.actor import SquashedDeterministicActor
from flowrl.module.critic import EnsembleCritic
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.module.rff import RffEnsembleCritic
from flowrl.types import Batch, Metric, Param, PRNGKey


@partial(jax.jit, static_argnames=("training", "num_samples", "solver"))
def jit_sample_actions(
    rng: PRNGKey,
    actor: Model,
    critic: Model,
    ddpm_target: Model,
    obs: jnp.ndarray,
    training: bool,
    num_samples: int,
    solver: str,
) -> Tuple[PRNGKey, jnp.ndarray]:
    assert len(obs.shape) == 2
    B = obs.shape[0]
    rng, xT_rng = jax.random.split(rng)

    # sample
    obs_repeat = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
    xT = jax.random.normal(xT_rng, (*obs_repeat.shape[:-1], actor.x_dim))
    rng, actions, _ = actor.sample(rng, xT, obs_repeat, training, solver)
    if num_samples == 1:
        actions = actions[:, 0]
    else:
        # t0 = jnp.zeros((obs_repeat.shape[0], num_samples, 1))
        # f0 = nce_target(obs_repeat, actions, t0, method="forward_phi")
        # qs = critic(obs_repeat, actions)
        t1 = jnp.ones((obs_repeat.shape[0], num_samples, 1), dtype=jnp.int32)
        f1 = ddpm_target(obs_repeat, actions, t1, method="forward_phi")
        qs = critic(f1).min(axis=0).reshape(B, num_samples)
        # qs = qs.min(axis=0).reshape(B, num_samples)
        best_idx = qs.argmax(axis=-1)
        actions = actions.reshape(B, num_samples, -1)[jnp.arange(B), best_idx]
    return rng, actions


@partial(jax.jit, static_argnames=("discount", "target_policy_noise", "noise_clip"))
def update_critic(
    rng: PRNGKey,
    critic: Model,
    critic_target: Model,
    actor_target: Model,
    ddpm_target: Model,
    diffusion_actor: Model,
    diffusion_value: Model,
    batch: Batch,
    discount: float,
    target_policy_noise: float,
    noise_clip: float,
    critic_coef: float
) -> Tuple[PRNGKey, Model, Metric]:
    t1 = jnp.ones((batch.obs.shape[0], 1), dtype=jnp.int32) * jnp.int32(1)
    rng, sample_rng = jax.random.split(rng)
    noise = jax.random.normal(sample_rng, batch.action.shape) * target_policy_noise
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_action = jnp.clip(actor_target(batch.next_obs) + noise, -1.0, 1.0)

    next_feature = ddpm_target(batch.next_obs, next_action, t1, method="forward_phi")
    q_target = critic_target(next_feature).min(0)
    q_target = batch.reward + discount * (1 - batch.terminal) * q_target

    back_critic_grad = False
    if back_critic_grad:
        raise NotImplementedError("no back critic grad exists")

    feature = ddpm_target(batch.obs, batch.action, t1, method="forward_phi")

    def critic_loss_fn(critic_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        q_pred = critic.apply(
            {"params": critic_params},
            feature,
            rngs={"dropout": dropout_rng},
        )
        critic_loss = critic_coef * ((q_pred - q_target[jnp.newaxis, :])**2).sum(0).mean()
        return critic_loss, {
            "loss/critic_loss": critic_loss,
            "misc/q_mean": q_pred.mean(),
            "misc/reward": batch.reward.mean(),
        }

    new_critic, metrics = critic.apply_gradient(critic_loss_fn)

    new_value = []
    value_metrics = {}
    for i in range(len(diffusion_value)):
        rng, eps_rng = jax.random.split(rng)
        eps = jax.random.normal(eps_rng, batch.action.shape)
        t = jnp.ones((batch.obs.shape[0], 1), dtype=jnp.int32) * jnp.int32(i)
        at = jnp.sqrt(diffusion_actor.alpha_hats[t]) * batch.action + jnp.sqrt(1-diffusion_actor.alpha_hats[t]) * eps
        ft = ddpm_target(batch.obs, at, t, method="forward_phi")
        def value_loss_fn(value_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
            qt_pred = diffusion_value[i].apply(
                {"params": value_params},
                ft,
            )
            value_loss = ((qt_pred - q_target) ** 2).mean()
            return value_loss, {
                f"loss/value_{i}_loss": value_loss,
            }
        this_value, this_metrics = diffusion_value[i].apply_gradient(value_loss_fn)
        new_value.append(this_value)
        value_metrics.update(this_metrics)

    return rng, new_critic, new_value, {
        **metrics,
        **value_metrics,
    }


@jax.jit
def update_actor(
    rng: PRNGKey,
    actor: Model,
    ddpm_target: Model,
    critic: Model,
    diffusion_actor,
    diffusion_value,
    batch: Batch,
) -> Tuple[PRNGKey, Model, Metric]:
    t1 = jnp.ones((batch.obs.shape[0], 1), dtype=jnp.int32) * jnp.int32(1)
    def actor_loss_fn(
        actor_params: Param, dropout_rng: PRNGKey
    ) -> Tuple[jnp.ndarray, Metric]:
        new_action = actor.apply(
            {"params": actor_params},
            batch.obs,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        new_feature = ddpm_target(batch.obs, new_action, t1, method="forward_phi")
        q = critic(new_feature)
        actor_loss = - q.mean()

        return actor_loss, {
            "loss/actor_loss": actor_loss,
        }

    new_actor, metrics = actor.apply_gradient(actor_loss_fn)

    # rng, at, t, eps = diffusion_actor.add_noise(rng, batch.action)
    # sigma = jnp.sqrt(1 - diffusion_actor.alpha_hats[t])
    rng, rng2 = jax.random.split(rng)
    B, S = batch.obs.shape
    A = batch.action.shape[-1]
    t_repeat = jnp.arange(diffusion_actor.steps+1)[..., jnp.newaxis, jnp.newaxis].repeat(B, axis=1)
    obs_repeat = batch.obs[jnp.newaxis, ...].repeat(diffusion_actor.steps+1, axis=0)
    action_repeat = batch.action[jnp.newaxis, ...].repeat(diffusion_actor.steps+1, axis=0)
    at_repeat = jnp.sqrt(diffusion_actor.alpha_hats[t_repeat]) * action_repeat + jnp.sqrt(1 - diffusion_actor.alpha_hats[t_repeat]) * jax.random.normal(rng2, action_repeat.shape)
    q_grad = []
    for i in range(diffusion_actor.steps+1):
        def get_q_value(at, obs, t):
            ft = ddpm_target(obs, at, t, method="forward_phi")
            q = diffusion_value[i](ft)
            return q.mean()
        q_grad_fn = jax.vmap(jax.grad(get_q_value))
        q_grad.append(q_grad_fn(at_repeat[i], obs_repeat[i], t_repeat[i]))
    q_grad = jnp.stack(q_grad, axis=0)
    # eps_estimation = -jnp.sqrt(1 - diffusion_actor.alpha_hats[t_repeat]) * q_grad
    # eps_estimation = l2_normalize(eps_estimation) * (eps_estimation.shape[-1] ** 0.5)
    # eps_estimation = eps_estimation / 0.2
    eps_estimation = - jnp.sqrt(1 - diffusion_actor.alpha_hats[t_repeat]) * q_grad / 0.1 / (jnp.abs(q_grad).mean() + 1e-6)

    def diffusion_loss_fn(diffusion_params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = diffusion_actor.apply(
            {"params": diffusion_params},
            at_repeat,
            t_repeat,
            condition=obs_repeat,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((eps_pred - eps_estimation) ** 2).mean()
        return loss, {
            "loss/diffusion_loss": loss,
            "misc/eps_estimation_l1": jnp.abs(eps_estimation).mean(),
        }
    new_diffusion, diffusion_metrics = diffusion_actor.apply_gradient(diffusion_loss_fn)
    return rng, new_actor, new_diffusion, {
        **metrics,
        **diffusion_metrics,
    }

@jax.jit
def jit_compute_metrics(
    rng: PRNGKey,
    critic: Model,
    ddpm_target: Model,
    diffusion_value: Model,
    diffusion_actor: Model,
    batch: Batch,
) -> Tuple[PRNGKey, Metric]:
    B, S = batch.obs.shape
    A = batch.action.shape[-1]
    num_actions = 50
    metrics = {}
    rng, action_rng = jax.random.split(rng)
    obs_repeat = batch.obs[..., jnp.newaxis, :].repeat(num_actions, axis=-2)
    action_repeat = jax.random.uniform(action_rng, (B, num_actions, A), minval=-1.0, maxval=1.0)

    def get_critic(at, obs):
        t1 = jnp.ones((1, ), dtype=jnp.int32)
        ft = ddpm_target(obs, at, t1, method="forward_phi")
        q = critic(ft)
        return q.mean()
    all_critic, all_critic_grad = jax.vmap(jax.value_and_grad(get_critic))(
        action_repeat.reshape(-1, A),
        obs_repeat.reshape(-1, S),
    )
    all_critic = all_critic.reshape(B, num_actions, 1)
    all_critic_grad = all_critic_grad.reshape(B, num_actions, -1)
    metrics.update({
        f"q_std/critic": all_critic.std(axis=1).mean(),
        f"q_grad/critic": jnp.abs(all_critic_grad).mean(),
    })

    for i in [0] + list(range(1, diffusion_actor.steps+1, diffusion_actor.steps//5)):
        def get_value(at, obs, t):
            ft = ddpm_target(obs, at, t, method="forward_phi")
            q = diffusion_value[i](ft)
            return q.mean()
        t_repeat = jnp.ones((B, num_actions, 1)) * jnp.int32(i)
        all_value, all_value_grad = jax.vmap(jax.value_and_grad(get_value))(
            action_repeat.reshape(-1, A),
            obs_repeat.reshape(-1, S),
            t_repeat.reshape(-1, 1),
        )
        all_value = all_value.reshape(B, num_actions, 1)
        all_value_grad = all_value_grad.reshape(B, num_actions, -1)
        metrics.update({
            f"q_std/value_{i}": all_value.std(axis=1).mean(),
            f"q_grad/value_{i}": jnp.abs(all_value_grad).mean(),
        })
    return rng, metrics

class DiffSRACASepAgent(TD3Agent):
    """
    Diff-SR with ACA agent.
    """

    name = "DiffSRACAAgent"
    model_names = ["ddpm", "ddpm_target", "actor", "actor_target", "critic", "critic_target"]

    def __init__(self, obs_dim: int, act_dim: int, cfg: DiffSRTD3Config, seed: int):
        super().__init__(obs_dim, act_dim, cfg, seed)
        self.cfg = cfg

        self.ddpm_coef = cfg.ddpm_coef
        self.critic_coef = cfg.critic_coef
        self.reward_coef = cfg.reward_coef
        self.num_noises = cfg.num_noises
        self.feature_dim = cfg.feature_dim

        # networks
        self.rng, ddpm_rng, ddpm_init_rng, actor_rng, critic_rng = jax.random.split(self.rng, 5)
        ddpm_def = FactorizedDDPM(
            self.obs_dim,
            self.act_dim,
            self.feature_dim,
            cfg.embed_dim,
            cfg.phi_hidden_dims,
            cfg.mu_hidden_dims,
            cfg.reward_hidden_dims,
            cfg.rff_dim,
            cfg.num_noises,
        )
        self.ddpm = Model.create(
            ddpm_def,
            ddpm_rng,
            inputs=(
                ddpm_init_rng,
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
            optimizer=optax.adam(learning_rate=cfg.feature_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.ddpm_target = Model.create(
            ddpm_def,
            ddpm_rng,
            inputs=(
                ddpm_init_rng,
                jnp.ones((1, self.obs_dim)),
                jnp.ones((1, self.act_dim)),
                jnp.ones((1, self.obs_dim)),
            ),
        )

        actor_def = SquashedDeterministicActor(
            backbone=MLP(
                hidden_dims=cfg.actor_hidden_dims,
                layer_norm=cfg.layer_norm,
                dropout=None,
            ),
            obs_dim=self.obs_dim,
            action_dim=self.act_dim,
        )
        # critic_def = RffEnsembleCritic(
        #     feature_dim=self.feature_dim,
        #     hidden_dims=cfg.critic_hidden_dims,
        #     rff_dim=cfg.rff_dim,
        #     ensemble_size=2,
        # )
        critic_def = EnsembleCritic(
            hidden_dims=[512, 512, 512],
            activation=jax.nn.elu,
            layer_norm=True,
            ensemble_size=2,
        )
        self.actor = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
            optimizer=optax.adam(learning_rate=cfg.actor_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.critic = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
            optimizer=optax.adam(learning_rate=cfg.critic_lr),
            clip_grad_norm=cfg.clip_grad_norm,
        )
        self.actor_target = Model.create(
            actor_def,
            actor_rng,
            inputs=(jnp.ones((1, self.obs_dim)),),
        )
        self.critic_target = Model.create(
            critic_def,
            critic_rng,
            inputs=(jnp.ones((1, self.feature_dim)),),
        )

        # define the ACA value and actor
        from flowrl.agent.online.unirep.diffsr.network import ResidualCritic
        from flowrl.flow.ddpm import DDPM, DDPMBackbone
        from flowrl.functional.activation import mish
        from flowrl.module.critic import Critic
        from flowrl.module.time_embedding import PositionalEmbedding
        self.rng, diffusion_value_rng, diffusion_actor_rng = jax.random.split(self.rng, 3)
        time_embedding = partial(PositionalEmbedding, output_dim=cfg.diffusion.time_dim)
        cond_embedding = partial(MLP, hidden_dims=(128, 128), activation=mish)
        noise_predictor = partial(
            MLP,
            hidden_dims=cfg.diffusion.mlp_hidden_dims,
            output_dim=act_dim,
            activation=mish,
            layer_norm=False,
            dropout=None,
        )
        backbone_def = DDPMBackbone(
            noise_predictor=noise_predictor,
            time_embedding=time_embedding,
            cond_embedding=cond_embedding,
        )
        self.diffusion_actor = DDPM.create(
            network=backbone_def,
            rng=diffusion_actor_rng,
            inputs=(jnp.ones((1, self.act_dim)), jnp.zeros((1, 1)), jnp.ones((1, self.obs_dim)), ),
            x_dim=self.act_dim,
            steps=cfg.diffusion.steps,
            noise_schedule="vp",
            noise_schedule_params={},
            approx_postvar=False,
            clip_sampler=cfg.diffusion.clip_sampler,
            x_min=cfg.diffusion.x_min,
            x_max=cfg.diffusion.x_max,
            optimizer=optax.adam(learning_rate=cfg.diffusion.lr),
        )
        value_def = Critic(
            hidden_dims=[512, 512, 512],
            activation=jax.nn.elu,
            layer_norm=True,
        )
        # value_def = ResidualCritic(
        #     time_embedding=time_embedding,
        #     hidden_dims=[512, 512],
        #     activation=jax.nn.elu,
        # )
        self.diffusion_value = []
        for t in range(cfg.diffusion.steps+1):
            this_rng, diffusion_value_rng = jax.random.split(diffusion_value_rng)
            self.diffusion_value.append(
                Model.create(
                    value_def,
                    this_rng,
                    inputs=(jnp.ones((1, self.feature_dim)), ),
                    optimizer=optax.adam(learning_rate=cfg.diffusion.lr),
                )
            )

        self._n_training_steps = 0

    def train_step(self, batch: Batch, step: int) -> Metric:
        metrics = {}

        self.rng, self.ddpm, ddpm_metrics = update_factorized_ddpm(
            self.rng,
            self.ddpm,
            batch,
            self.reward_coef,
        )
        metrics.update(ddpm_metrics)

        self.rng, self.critic, self.diffusion_value, critic_metrics = update_critic(
            self.rng,
            self.critic,
            self.critic_target,
            self.actor_target,
            self.ddpm_target,
            self.diffusion_actor,
            self.diffusion_value,
            batch,
            discount=self.cfg.discount,
            target_policy_noise=self.target_policy_noise,
            noise_clip=self.noise_clip,
            critic_coef=self.critic_coef,
        )
        metrics.update(critic_metrics)

        if self._n_training_steps % self.actor_update_freq == 0:
            self.rng, self.actor, self.diffusion_actor, actor_metrics = update_actor(
                self.rng,
                self.actor,
                self.ddpm_target,
                self.critic,
                self.diffusion_actor,
                self.diffusion_value,
                batch,
            )
            metrics.update(actor_metrics)

        if self._n_training_steps % self.target_update_freq == 0:
            self.sync_target()

        if self._n_training_steps % 2000 == 0:
            self.rng, metrics = jit_compute_metrics(
                self.rng,
                self.critic,
                self.ddpm_target,
                self.diffusion_value,
                self.diffusion_actor,
                batch,
            )
            metrics.update(metrics)
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
            self.rng, action = jit_sample_actions(
                self.rng,
                self.diffusion_actor,
                self.critic,
                self.ddpm_target,
                obs,
                training=False,
                num_samples=num_samples,
                solver=self.cfg.diffusion.solver,
            )
        else:
            action = super().sample_actions(obs, deterministic, num_samples)
        return action, {}

    def sync_target(self):
        super().sync_target()
        self.ddpm_target = ema_update(self.ddpm, self.ddpm_target, self.cfg.feature_ema)
