import inspect
from functools import partial

import jax
import jax.numpy as jnp

from flowrl.types import *

sg = lambda x: jax.lax.stop_gradient(x)


def fused_update(
    func: Callable,
    num_updates: int,
    static_argnames: Optional[Union[Sequence[str], str]] = None,
):
    """
    Decorator that fuses multiple calls to a function into a single jitted function.

    The decorator splits the first argument (treated as the batch) into mini-batches
    and loops through num_updates iterations, calling the original function on each
    mini-batch. All static arguments from the original function are preserved.

    Assumes that return values (except the last one, typically metrics) correspond
    to input arguments (excluding the first batch arg and static args) in the same order.
    That is, the first N-1 return values update the first N non-static, non-batch
    input arguments for the next iteration.

    Args:
        func: The function to decorate (may already be jitted)
        num_updates: Number of times to call the function (fused into one loop)
        static_argnames: Static argument names from the original function.
                        If None, uses empty tuple.
                        Can be a sequence of strings or a comma-separated string.

    Returns:
        A new jitted function that fuses num_updates calls to the original function.
        The new function has the same signature as the original, with 'num_updates'
        added as a static argument internally.

    Example:
        @partial(jax.jit, static_argnames=("discount", "ema", "target_entropy"))
        def update_once(batch, rng, actor, critic, discount, ema, target_entropy):
            return rng, actor, critic, metrics

        update_multiple = fuse_updates(
            update_once,
            num_updates=4,
            static_argnames=("discount", "ema", "target_entropy")
        )
        # update_multiple is jitted with static_argnames=("num_updates", "discount", "ema", "target_entropy")
    """
    # Get the original function (unwrap if it's already jitted)
    original_func = func
    while hasattr(original_func, '__wrapped__'):
        original_func = original_func.__wrapped__

    # Normalize static_argnames
    if static_argnames is None:
        static_argnames = ()
    elif isinstance(static_argnames, str):
        static_argnames = tuple(name.strip() for name in static_argnames.split(','))
    else:
        static_argnames = tuple(static_argnames)

    # Get function signature
    sig = inspect.signature(original_func)
    param_names = list(sig.parameters.keys())

    # Use first parameter as batch argument
    if not param_names:
        raise ValueError("Function must have at least one parameter")
    batch_arg_name = param_names[0]

    # Identify non-static, non-batch arguments (these will be updated from return values)
    non_static_non_batch_indices = [
        i for i, name in enumerate(param_names)
        if name != batch_arg_name and name not in static_argnames
    ]

    # Create the fused update function
    @partial(jax.jit, static_argnames=("num_updates",) + static_argnames)
    def fused_update(num_updates: int, *args, **kwargs):
        # Combine args and kwargs into a single dict
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arg_dict = bound.arguments

        # Extract batch argument
        batch = arg_dict[batch_arg_name]

        # Split batch into mini-batches
        if hasattr(batch, 'obs'):  # Handle Batch dataclass
            mini_batch_size = batch.obs.shape[0] // num_updates
            batch_split = jax.tree.map(
                lambda x: x.reshape((num_updates, mini_batch_size, -1)) if x is not None else None,
                batch
            )
        else:  # Handle generic array or pytree
            # Assume first dimension is batch dimension
            first_leaf = next(iter(jax.tree.leaves(batch)), batch)
            mini_batch_size = first_leaf.shape[0] // num_updates
            batch_split = jax.tree.map(
                lambda x: x.reshape((num_updates, mini_batch_size, *x.shape[1:])) if x is not None else None,
                batch
            )

        # Helper function for one update
        def one_update(i, state, first_iteration: bool = False):
            # state is the tuple of return values from previous iteration
            # For first iteration (i=0), state is empty and we use original args
            # For subsequent iterations, state contains updated values from previous call

            if first_iteration:
                # First iteration: use original arguments (except batch)
                mini_batch = jax.tree.map(
                    lambda x: jnp.take(x, i, axis=0) if x is not None else None,
                    batch_split
                )
                call_args = [
                    arg_dict[name] if name != batch_arg_name else mini_batch
                    for name in param_names
                ]
                result = original_func(*call_args)
                return result if isinstance(result, tuple) else (result,)
            else:
                # Subsequent iterations: use updated values from previous call
                prev_returns = state

                # Map return values to input arguments
                # Return values (except last, typically metrics) correspond to non-static, non-batch args
                num_updating_args = len(non_static_non_batch_indices)
                if len(prev_returns) > num_updating_args:
                    # Last return value is metrics, skip it for input mapping
                    updated_values = prev_returns[:-1]
                else:
                    updated_values = prev_returns

                # Create mapping from param index to updated value or original value
                value_map = {}
                for j, idx in enumerate(non_static_non_batch_indices):
                    if j < len(updated_values):
                        value_map[idx] = updated_values[j]

                # Get mini-batch for this iteration
                mini_batch = jax.tree.map(
                    lambda x: jnp.take(x, i, axis=0) if x is not None else None,
                    batch_split
                )

                # Build call arguments: use updated values if available, otherwise original args
                call_args = []
                for idx, name in enumerate(param_names):
                    if name == batch_arg_name:
                        call_args.append(mini_batch)
                    elif idx in value_map:
                        call_args.append(value_map[idx])
                    else:
                        call_args.append(arg_dict[name])

                result = original_func(*call_args)
                return result if isinstance(result, tuple) else (result,)

        # First update to initialize state
        initial_state = one_update(0, (), first_iteration=True)

        # Loop through remaining updates
        final_state = jax.lax.fori_loop(1, num_updates, one_update, initial_state)

        return final_state

    # Create wrapper that injects num_updates
    def wrapper(*args, **kwargs):
        return fused_update(num_updates, *args, **kwargs)

    # Copy function metadata
    wrapper.__name__ = f"fused_{original_func.__name__}"
    wrapper.__doc__ = f"Fused version of {original_func.__name__} with {num_updates} updates.\n\n{original_func.__doc__ or ''}"

    return wrapper
