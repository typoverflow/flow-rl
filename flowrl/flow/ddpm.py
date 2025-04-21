from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.struct import PyTreeNode, dataclass, field
from flax.training.train_state import TrainState

from flowrl.functional.activation import mish
from flowrl.functional.ema import ema_update
from flowrl.module.mlp import MLP
from flowrl.module.model import Model
from flowrl.types import *

SUPPORTED_SOLVERS = [
    "ddim", "ddpm",
]


# ======= Noise Schedules ========

def linear_beta_schedule(T: int=1000, beta_min: float=1e-4, beta_max: float=2e-2):
    return jnp.linspace(beta_min, beta_max, T)

def cosine_beta_schedule(T: int=1000, s=0.008):
    steps = T + 1
    t = jnp.linspace(0, T, steps) / T
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)

def vp_beta_schedule(T: int=1000):
    t = jnp.arange(1, T + 1)
    b_max = 10.
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas

# ======= Noise Network ========

class FourierEmbedding(nn.Module):
    output_dim: int = 16
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        if self.learnable:
            w = self.param('kernel', nn.initializers.normal(0.2),
                           (self.output_dim // 2, x.shape[-1]), jnp.float32)
            f = 2 * jnp.pi * x @ w.T
        else:
            half_dim = self.output_dim // 2
            f = jnp.log(10000) / (half_dim - 1)
            f = jnp.exp(jnp.arange(half_dim) * -f)
            f = x * f
        return jnp.concatenate([jnp.cos(f), jnp.sin(f)], axis=-1)


class DDPMBackbone(nn.Module):
    noise_predictor: nn.Module
    time_embedding: nn.Module
    cond_embedding: nn.Module = None

    @nn.compact
    def __call__(
        self,
        s: jnp.ndarray,  # s = [obs, cond] if it's conditional
        a: jnp.ndarray,
        time: jnp.ndarray,
        training: bool = False
    ):
        t_ff = self.time_embedding()(time)
        t_ff = MLP(
            hidden_dims=[t_ff.shape[-1], t_ff.shape[-1]],
            activation=mish,
        )(t_ff)
        if self.cond_embedding is not None:
            # last dim gives the class token
            embed_feature = self.cond_embedding()(s[:, -1])  # gives the shape of (B,) array
            s = jnp.concatenate([s[:, :-1], embed_feature], axis=-1)

        input = jnp.concatenate([s, a, t_ff], axis=-1)
        return self.noise_predictor()(input, training=training)


class CriticT(nn.Module):
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None

    @nn.compact
    def __call__(
        self,
        obs: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        t_ff = self.time_embedding()(t)
        t_ff = MLP(
            hidden_dims=[t_ff.shape[-1], t_ff.shape[-1]],
            activation=mish,
        )(t_ff)
        x = jnp.concatenate([item for item in [obs, action, t_ff] if item is not None], axis=-1)
        x = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout,
        )(x, training)
        return x


class EnsembleCriticT(nn.Module):
    time_embedding: nn.Module
    hidden_dims: Sequence[int]
    activation: Callable = nn.relu
    layer_norm: bool = False
    dropout: Optional[float] = None
    ensemble_size: int = 2

    @nn.compact
    def __call__(
        self,
        obs: Optional[jnp.ndarray] = None,
        action: Optional[jnp.ndarray] = None,
        t: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        vmap_critic_t = nn.vmap(
            CriticT,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size
        )
        x = vmap_critic_t(
            time_embedding=self.time_embedding,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout=self.dropout
        )(obs, action, t, training)
        return x

# ======= DDPM ========

@dataclass
class DDPM(Model):
    state: TrainState
    dropout_rng: PRNGKey = field(pytree_node=True)
    x_dim: int = field(pytree_node=False, default=None)
    steps: int = field(pytree_node=False, default=None)
    clip_sampler: bool = field(pytree_node=False, default=None)
    x_min: float = field(pytree_node=False, default=None)
    x_max: float = field(pytree_node=False, default=None)
    betas: jnp.ndarray = field(pytree_node=True, default=None)
    alphas: jnp.ndarray = field(pytree_node=True, default=None)
    alpha_hats: jnp.ndarray = field(pytree_node=True, default=None)

    @classmethod
    def create(
        cls,
        network: nn.Module,
        rng: PRNGKey,
        inputs: Sequence[jnp.ndarray],
        x_dim: int,
        steps: int,
        noise_schedule: str,
        noise_schedule_params: Optional[Dict]=None,
        clip_sampler: bool=False,
        x_min: Optional[float]=None,
        x_max: Optional[float]=None,
        optimizer: Optional[optax.GradientTransformation]=None,
        clip_grad_norm: float=None
    ) -> 'DDPM':
        ret = super().create(network, rng, inputs, optimizer, clip_grad_norm)

        if noise_schedule_params is None:
            noise_schedule_params = {}
        noise_schedule_params["T"] = steps
        if noise_schedule == "linear":
            betas = linear_beta_schedule(**noise_schedule_params)
        elif noise_schedule == "cosine":
            betas = cosine_beta_schedule(**noise_schedule_params)
        elif noise_schedule == "vp":
            betas = vp_beta_schedule(**noise_schedule_params)
        else:
            raise NotImplementedError(f"Unsupported noise schedule: {noise_schedule}")
        betas = jnp.concatenate([jnp.zeros((1, )), betas])
        alphas = 1 - betas
        alpha_hats = jnp.cumprod(alphas)

        return ret.replace(
            x_dim=x_dim,
            steps=steps,
            clip_sampler=clip_sampler,
            x_min=x_min,
            x_max=x_max,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hats
        )

    def add_noise(self, rng, x0):
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (*x0.shape[:-1], 1), 1, self.steps+1)
        eps = jax.random.normal(noise_rng, x0.shape)

        xt = jnp.sqrt(self.alpha_hats[t]) * x0 + jnp.sqrt(1-self.alpha_hats[t]) * eps
        return rng, xt, t, eps

    @partial(jax.jit, static_argnames=("training", "num_samples", "solver"))
    def sample(
        self,
        rng: PRNGKey,
        obs: jnp.ndarray,
        training: bool,
        num_samples: Optional[int],
        solver: str="ddpm",
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:

        rng, xT_rng, dropout_rng = jax.random.split(rng, 3)
        if num_samples is not None:
            obs_use = obs[..., jnp.newaxis, :].repeat(num_samples, axis=-2)
        else:
            obs_use = obs
        xT = jax.random.normal(xT_rng, (*obs_use.shape[:-1], self.x_dim))
        t_proto = jnp.ones((*obs.shape[:-1], 1), dtype=jnp.int32)

        def fn(input, t):
            rng_, xt = input
            input_t = t_proto * t

            if training:
                eps_theta = self.apply(
                    {"params": params}, obs_use, xt, input_t, training=training, rngs={"dropout": dropout_rng}
                )
            else:
                eps_theta = self(obs_use, xt, input_t, training=training)

            if solver == "ddpm":
                x0_hat = 1 / jnp.sqrt(self.alpha_hats[t]) * (xt - jnp.sqrt(1 - self.alpha_hats[t]) * eps_theta)
                if self.clip_sampler:
                    x0_hat = jnp.clip(x0_hat, self.x_min, self.x_max)
                xt_1 = 1 / (1 - self.alpha_hats[t]) * (jnp.sqrt(self.alpha_hats[t - 1]) * (1 - self.alphas[t]) * x0_hat +
                        jnp.sqrt(self.alphas[t]) * (1 - self.alpha_hats[t - 1]) * xt)
                rng_, key_ = jax.random.split(rng_)
                std_t = jnp.sqrt(1-self.alphas[t])
                xt_1 += (t>1) * std_t * jax.random.normal(key_, xt.shape)
            elif solver == "ddim":
                alpha_1 = 1 / jnp.sqrt(self.alphas[t])
                alpha_2 = jnp.sqrt(1 - self.alpha_hats[t])
                alpha_3 = jnp.sqrt(1 - self.alpha_hats[t-1])
                xt_1 = alpha_1 * (xt - alpha_2 * eps_theta) + alpha_3 * eps_theta
                if self.clip_sampler:
                    xt_1 = jnp.clip(xt_1, -self.x_min, self.x_max)
            else:
                raise NotImplementedError(f"Unsupported solver {solver} for {type(self)}")

            return (rng_, xt_1), (xt, eps_theta)

        output, history = jax.lax.scan(fn, (rng, xT), jnp.arange(self.steps, 0, -1), unroll=True)
        rng, action = output
        return rng, action, history

    @partial(jax.jit, static_argnames=("training", "num_samples", "solver", "sample_xt"))
    def onestep_sample(
        self,
        rng: PRNGKey,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        training: bool,
        num_samples: int,
        solver: str="ddpm",
        sample_xt: bool=True,
        t: Optional[jnp.ndarray]=None,
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        assert num_samples is not None
        B = obs.shape[0]
        rng, sample_rng, noise_rng = jax.random.split(rng, 3)
        if sample_xt:
            x0 = action
            _, xt, t, eps_sample = self.add_noise(sample_rng, x0)
        else:
            t = t
            xt = action
        t_1 = t - 1
        repeated_t = t.repeat(num_samples, axis=0).reshape(B, num_samples, -1)
        repeated_t_1 = t_1.repeat(num_samples, axis=0).reshape(B, num_samples, -1)

        if training: # BUG: dropout_rng is not passed
            eps_theta = self.apply(
                {"params": params}, obs, xt, t, training=training
            )
        else:
            eps_theta = self(obs, xt, t, training=training)

        if solver == "ddpm":
            x0_hat = 1 / jnp.sqrt(self.alpha_hats[t]) * (xt - jnp.sqrt(1 - self.alpha_hats[t]) * eps_theta)
            if self.clip_sampler:
                x0_hat = jnp.clip(x0_hat, self.x_min, self.x_max)
            xt_1 = 1 / (1 - self.alpha_hats[t]) * (jnp.sqrt(self.alpha_hats[t - 1]) * (1 - self.alphas[t]) * x0_hat +
                    jnp.sqrt(self.alphas[t]) * (1 - self.alpha_hats[t - 1]) * xt)
            repeated_xt_1 = xt_1.repeat(num_samples, axis=0).reshape(B, num_samples, -1)
            noise = jax.random.normal(noise_rng, repeated_xt_1.shape)
            std_t = jnp.sqrt(1-self.alphas[repeated_t])
            repeated_xt_1 += (repeated_t > 1) * std_t * noise
        elif solver == "ddim":
            alpha_1 = 1 / jnp.sqrt(self.alphas[t])
            alpha_2 = jnp.sqrt(1 - self.alpha_hats[t])
            alpha_3 = jnp.sqrt(1 - self.alpha_hats[t-1])
            xt_1 = alpha_1 * (xt - alpha_2 * eps_theta) + alpha_3 * eps_theta
            repeated_xt_1 = xt_1.repeat(num_samples, axis=0).reshape(B, num_samples, -1)
            if self.clip_sampler:
                repeated_xt_1 = jnp.clip(repeated_xt_1, self.x_min, self.x_max)
        else:
            raise NotImplementedError(f"Unsupported solver {solver} for {type(self)}")

        return rng, repeated_xt_1, xt, repeated_t_1, t, eps_theta

# ======= Update Function ========

@jax.jit
def jit_update_ddpm(
    rng: PRNGKey,
    model: DDPM,
    batch: Batch,
) -> Tuple[PRNGKey, DDPM, Metric]:
    x0 = batch.action
    rng, xt, t, eps = model.add_noise(rng, x0)

    def loss_fn(params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = model.apply(
            {"params": params},
            batch.obs,
            xt,
            t,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((eps_pred - eps) ** 2).mean()
        return loss, {
            "ddpm_loss": loss
        }

    new_model, metrics = model.apply_gradient(loss_fn)
    return rng, new_model, metrics
