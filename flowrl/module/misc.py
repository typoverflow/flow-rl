import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


class TunableCoefficient(nn.Module):
    init_value: float | None = None

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        if self.init_value is None:
            return self.param("value", lambda key: nn.initializers.normal()(key, (), jnp.float32))
        else:
            return self.param("value", lambda key: jnp.full((), self.init_value, dtype=jnp.float32))


class PositiveTunableCoefficient(nn.Module):
    init_value: float | None = None

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        assert self.init_value > 0, f"init_value must be positive, got {self.init_value}"

        if self.init_value is None:
            log_value = self.param("value", lambda key: nn.initializers.normal()(key, (), jnp.float32))
        else:
            log_value = self.param("value", lambda key: jnp.full((), jnp.log(self.init_value), dtype=jnp.float32))
        return jnp.exp(log_value)
