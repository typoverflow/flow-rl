import jax

sg = lambda x: jax.lax.stop_gradient(x)
