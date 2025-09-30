import jax.numpy as jnp


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
):
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
