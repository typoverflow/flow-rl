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

def linear_beta_schedule(T: int = 1000, beta_min: float = 1e-4, beta_max: float = 2e-2):
    return jnp.linspace(beta_min, beta_max, T)


def cosine_beta_schedule(T: int = 1000, s=0.008):
    steps = T + 1
    t = jnp.linspace(0, T, steps) / T
    alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0, 0.999)


def vp_beta_schedule(T: int = 1000):
    t = jnp.arange(1, T + 1)
    b_max = 10.0
    b_min = 0.1
    alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    betas = 1 - alpha
    return betas


def get_noise_schedule(
    noise_schedule: str = "linear",
    num_noises: int = 1000,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    s: float = 0.008,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if noise_schedule == "linear":
        betas = linear_beta_schedule(num_noises, beta_min, beta_max)
    elif noise_schedule == "cosine":
        betas = cosine_beta_schedule(num_noises, s)
    elif noise_schedule == "vp":
        betas = vp_beta_schedule(num_noises)
    else:
        raise NotImplementedError
    alphas = 1 - betas
    alphabars = jnp.cumprod(alphas, axis=0)
    return (
        jnp.array(betas, dtype=jnp.float32),
        jnp.array(alphas, dtype=jnp.float32),
        jnp.array(alphabars, dtype=jnp.float32),
    )

# ======= Noise Network ========

class DDPMBackbone(nn.Module):
    noise_predictor: nn.Module
    time_embedding: nn.Module = None
    cond_embedding: nn.Module = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        time: jnp.ndarray,
        condition: Optional[jnp.ndarray] = None,
        training: bool = False
    ):
        if self.time_embedding is not None:
            time = self.time_embedding()(time)
            time = MLP(
                hidden_dims=[time.shape[-1], time.shape[-1]],
                activation=mish,
            )(time)
        if self.cond_embedding is not None:
            condition = self.cond_embedding()(condition, training=training)
        if condition is not None:
            inputs = jnp.concatenate([x, time, condition], axis=-1)
        else:
            inputs = jnp.concatenate([x, time], axis=-1)
        return self.noise_predictor()(inputs, training=training)

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
    postvars: jnp.ndarray = field(pytree_node=True, default=None)

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
        approx_postvar: bool=False,
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
        if approx_postvar:
            postvars = betas
        else:
            postvars = betas[1:] * (1-alpha_hats[:-1]) / (1-alpha_hats[1:])
            postvars = jnp.concatenate([jnp.zeros((1, )), postvars])

        return ret.replace(
            x_dim=x_dim,
            steps=steps,
            clip_sampler=clip_sampler,
            x_min=x_min,
            x_max=x_max,
            betas=betas,
            alphas=alphas,
            alpha_hats=alpha_hats,
            postvars=postvars,
        )

    @partial(jax.jit, static_argnames=())
    def add_noise(self, rng, x0):
        rng, t_rng, noise_rng = jax.random.split(rng, 3)
        t = jax.random.randint(t_rng, (*x0.shape[:-1], 1), 1, self.steps+1)
        eps = jax.random.normal(noise_rng, x0.shape)

        xt = jnp.sqrt(self.alpha_hats[t]) * x0 + jnp.sqrt(1-self.alpha_hats[t]) * eps
        return rng, xt, t, eps

    @partial(jax.jit, static_argnames=("training", "solver"))
    def sample(
        self,
        rng: PRNGKey,
        xT: jnp.ndarray,
        condition: Optional[jnp.ndarray]=None,
        training: bool=False,
        solver: str="ddpm",
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:

        t_proto = jnp.ones((*xT.shape[:-1], 1), dtype=jnp.int32)

        def fn(input, t):
            rng_, xt = input
            rng_, dropout_rng_ = jax.random.split(rng_)
            input_t = t_proto * t

            if training:
                eps_theta = self.apply(
                    {"params": params}, xt, input_t, condition=condition, training=training, rngs={"dropout": dropout_rng_}
                )
            else:
                eps_theta = self(xt, input_t, condition=condition, training=training)

            if solver == "ddpm":
                x0_hat = 1 / jnp.sqrt(self.alpha_hats[t]) * (xt - jnp.sqrt(1 - self.alpha_hats[t]) * eps_theta)
                if self.clip_sampler:
                    x0_hat = jnp.clip(x0_hat, self.x_min, self.x_max)
                xt_1 = 1 / (1 - self.alpha_hats[t]) * (jnp.sqrt(self.alpha_hats[t - 1]) * (1 - self.alphas[t]) * x0_hat +
                        jnp.sqrt(self.alphas[t]) * (1 - self.alpha_hats[t - 1]) * xt)
                rng_, key_ = jax.random.split(rng_)
                std_t = jnp.sqrt(self.postvars[t])
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
        xt_or_x0: jnp.ndarray,
        condition: Optional[jnp.ndarray]=None,
        training: bool=False,
        num_samples: Optional[int]=None,
        solver: str="ddpm",
        sample_xt: bool=True,
        t: Optional[jnp.ndarray]=None,
        params: Optional[Param]=None,
    ) -> Tuple[PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B = xt_or_x0.shape[0]
        rng, sample_rng, noise_rng, dropout_rng = jax.random.split(rng, 4)
        if sample_xt:
            x0 = xt_or_x0
            _, xt, t, eps_sample = self.add_noise(sample_rng, x0)
        else:
            t = t
            xt = xt_or_x0
        t_1 = t - 1
        repeated_t = t.repeat(num_samples, axis=0).reshape(B, num_samples, -1)
        repeated_t_1 = t_1.repeat(num_samples, axis=0).reshape(B, num_samples, -1)

        if training:
            eps_theta = self.apply(
                {"params": params}, xt, t, condition=condition, training=training, rngs={"dropout": dropout_rng}
            )
        else:
            eps_theta = self(xt, t, condition=condition, training=training)

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
    x0: jnp.ndarray,
    condition: Optional[jnp.ndarray]=None,
) -> Tuple[PRNGKey, DDPM, Metric]:
    rng, xt, t, eps = model.add_noise(rng, x0)

    def loss_fn(params: Param, dropout_rng: PRNGKey) -> Tuple[jnp.ndarray, Metric]:
        eps_pred = model.apply(
            {"params": params},
            xt,
            t,
            condition=condition,
            training=True,
            rngs={"dropout": dropout_rng},
        )
        loss = ((eps_pred - eps) ** 2).mean()
        return loss, {
            "ddpm_loss": loss
        }

    new_model, metrics = model.apply_gradient(loss_fn)
    return rng, new_model, metrics
