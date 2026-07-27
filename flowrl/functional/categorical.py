import jax
import jax.numpy as jnp


def project_categorical(
    probabilities: jnp.ndarray,
    target_atoms: jnp.ndarray,
    v_min: float,
    v_max: float,
) -> jnp.ndarray:
    """Project categorical targets onto an evenly spaced fixed support."""
    num_atoms = probabilities.shape[-1]
    delta = (v_max - v_min) / (num_atoms - 1)
    target_atoms = jnp.clip(target_atoms, v_min, v_max)
    locations = (target_atoms - v_min) / delta
    lower = jnp.clip(
        jnp.floor(locations).astype(jnp.int32), 0, num_atoms - 1
    )
    upper = jnp.clip(
        jnp.ceil(locations).astype(jnp.int32), 0, num_atoms - 1
    )
    equal = lower == upper
    lower_weight = jnp.where(equal, 1.0, upper - locations)
    upper_weight = jnp.where(equal, 0.0, locations - lower)

    lower_mass = (
        probabilities[..., None]
        * lower_weight[..., None]
        * jax.nn.one_hot(lower, num_atoms)
    ).sum(axis=-2)
    upper_mass = (
        probabilities[..., None]
        * upper_weight[..., None]
        * jax.nn.one_hot(upper, num_atoms)
    ).sum(axis=-2)
    return lower_mass + upper_mass
