import distrax
import jax
import jax.numpy as jnp

from flowrl.types import PRNGKey, Shape


class TanhMultivariateNormalDiag(distrax.Transformed):
    def __init__(self, loc: jax.Array, scale_diag: jax.Array) -> None:
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)
        super().__init__(
            distribution=distribution, bijector=distrax.Block(distrax.Tanh(), 1)
        )

    def _clip(self, sample: jnp.ndarray) -> jnp.ndarray:
        clip_bound = 1.0 - jnp.finfo(sample.dtype).eps
        return jnp.clip(sample, -clip_bound, clip_bound)

    def tanh_mean(self) -> jnp.ndarray:
        mean = jnp.tanh(self.distribution.mean())
        return self._clip(mean)

    def sample(
        self, *, seed: PRNGKey, sample_shape: Shape = () # type: ignore
    ) -> jnp.ndarray:
        sample = super().sample(seed=seed, sample_shape=sample_shape)
        return self._clip(sample)

    def log_prob(self, *args, **kwargs):
        return super().log_prob(*args, **kwargs)[..., jnp.newaxis]

    def sample_and_log_prob(
        self, *, seed: PRNGKey, sample_shape: Shape = () # type: ignore
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        sample, log_prob = super().sample_and_log_prob(
            seed=seed, sample_shape=sample_shape
        )
        return self._clip(sample), log_prob[..., jnp.newaxis]
