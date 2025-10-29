import jax
import jax.numpy as jnp


def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def two_hot(x, num_bins, vmin, vmax, bin_size):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if num_bins == 0:
		return x
	elif num_bins == 1:
		return symlog(x)
	x = jnp.clip(symlog(x), vmin, vmax).squeeze(1)
	bin_idx = jnp.floor((x - vmin) / bin_size)
	bin_offset = ((x - vmin) / bin_size - bin_idx)[..., jnp.newaxis]
	soft_two_hot = jnp.zeros((x.shape[0], num_bins), dtype=x.dtype)
	soft_two_hot = soft_two_hot.at[jnp.arange(x.shape[0]), bin_idx].set(1 - bin_offset)
	soft_two_hot = soft_two_hot.at[jnp.arange(x.shape[0]), (bin_idx + 1) % num_bins].set(bin_offset)
	return soft_two_hot


def two_hot_inv(x, num_bins, vmin, vmax, bin_size):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	if num_bins == 0:
		return x
	elif num_bins == 1:
		return symexp(x)
	dreg_bins = jnp.linspace(vmin, vmax, num_bins, dtype=x.dtype)
	x = jax.nn.softmax(x, axis=-1)
	x = jnp.sum(x * dreg_bins, axis=-1, keepdims=True)
	return symexp(x)
