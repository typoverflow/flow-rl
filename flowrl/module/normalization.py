import flax.linen as nn
import jax
import jax.numpy as jnp


class BatchRenorm(nn.Module):
    """Batch Renormalization with the schedule used by DIME/CrossQ."""

    momentum: float = 0.99
    epsilon: float = 1e-3
    warmup_steps: int = 100_000
    r_max: float = 3.0
    d_max: float = 5.0
    use_scale: bool = True
    use_bias: bool = True

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        training: bool = False,
    ) -> jnp.ndarray:
        feature_shape = (x.shape[-1],)
        running_mean = self.variable(
            "batch_stats", "mean", jnp.zeros, feature_shape
        )
        running_var = self.variable(
            "batch_stats", "var", jnp.ones, feature_shape
        )
        steps = self.variable(
            "batch_stats", "steps", lambda: jnp.array(0, dtype=jnp.int32)
        )

        if training:
            mean = jnp.mean(x, axis=tuple(range(x.ndim - 1)))
            var = jnp.var(x, axis=tuple(range(x.ndim - 1)))
            std = jnp.sqrt(var + self.epsilon)
            running_std = jnp.sqrt(running_var.value + self.epsilon)

            r = jax.lax.stop_gradient(
                jnp.clip(std / running_std, 1.0 / self.r_max, self.r_max)
            )
            d = jax.lax.stop_gradient(
                jnp.clip(
                    (mean - running_mean.value) / running_std,
                    -self.d_max,
                    self.d_max,
                )
            )
            warmed_up = steps.value >= self.warmup_steps
            renorm_var = var / jnp.square(r)
            renorm_mean = mean - d * jnp.sqrt(var) / r
            normalization_mean = jnp.where(
                warmed_up, renorm_mean, mean
            )
            normalization_var = jnp.where(
                warmed_up, renorm_var, var
            )
            x = (x - normalization_mean) / jnp.sqrt(
                normalization_var + self.epsilon
            )

            if not self.is_initializing():
                running_mean.value = (
                    self.momentum * running_mean.value
                    + (1 - self.momentum) * mean
                )
                running_var.value = (
                    self.momentum * running_var.value
                    + (1 - self.momentum) * var
                )
                steps.value += 1
        else:
            x = (x - running_mean.value) / jnp.sqrt(
                running_var.value + self.epsilon
            )

        if self.use_scale:
            scale = self.param("scale", nn.initializers.ones, feature_shape)
            x = x * scale
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, feature_shape)
            x = x + bias
        return x
