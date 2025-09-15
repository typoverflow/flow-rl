from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from flowrl.types import PRNGKey


class TreeState(NamedTuple):
    tree: jnp.ndarray
    pointer: int
    n_entries: int


class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.n_levels = int(jnp.ceil(jnp.log2(capacity)))
        self.state = TreeState(
            tree=jnp.zeros(2**(self.n_levels+1) - 1),
            pointer=0,
            n_entries=0,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("n_levels"))
    def _jitted_propagate_tree(tree, tree_idx, new_value, n_levels):
        change = new_value - tree[tree_idx]
        new_tree = tree.at[tree_idx].add(change)
        def body_fun(i, carry):
            cur_tree_idx, cur_tree = carry
            parent_idx = (cur_tree_idx - 1) // 2
            return (parent_idx, cur_tree.at[parent_idx].add(change))
        return jax.lax.fori_loop(0, n_levels, body_fun, (tree_idx, new_tree), unroll=True)[1]

    @staticmethod
    @partial(jax.jit, static_argnames=("n_levels", "capacity", "propagate_func"))
    def _jitted_add_batch(state, priorities, n_levels, capacity, propagate_func):
        def body_fun(i, loop_state):
            priority = priorities[i]
            tree_idx = loop_state.pointer + 2**n_levels - 1

            propagated_tree = propagate_func(
                loop_state.tree,
                tree_idx,
                priority,
                n_levels
            )

            new_pointer = (loop_state.pointer + 1) % capacity
            new_n_entries = jnp.minimum(loop_state.n_entries + 1, capacity)

            return loop_state._replace(
                tree=propagated_tree,
                pointer=new_pointer,
                n_entries=new_n_entries,
            )

        batch_size = priorities.shape[0]
        final_state = jax.lax.fori_loop(0, batch_size, body_fun, state, unroll=True)
        return final_state

    @staticmethod
    @partial(jax.jit, static_argnames=("n_levels", "propagate_func"))
    def _jitted_update_batch(state, priorities, indices, n_levels, propagate_func):
        def body_fun(i, loop_state):
            priority = priorities[i]
            index = indices[i]
            tree_idx = index + 2**n_levels - 1

            propagated_tree = propagate_func(
                loop_state.tree,
                tree_idx,
                priority,
                n_levels
            )

            return loop_state._replace(
                tree=propagated_tree,
            )

        batch_size = priorities.shape[0]
        final_state = jax.lax.fori_loop(0, batch_size, body_fun, state, unroll=True)
        return final_state

    @staticmethod
    @partial(jax.jit, static_argnames=("batch_size", "n_levels"))
    def _jitted_sample_batch(rng: PRNGKey, state: TreeState, batch_size: int, n_levels: int):
        target_values = jax.random.uniform(rng, shape=(batch_size,)) * state.tree[0]
        def _retrieve_single(target_val):
            def body_fun(i, carry):
                cur_tree_idx, cur_val = carry
                left_child = 2 * cur_tree_idx + 1
                left_val = state.tree[left_child]
                go_left = cur_val <= left_val
                next_tree_idx = jnp.where(go_left, left_child, left_child + 1)
                next_val = jnp.where(go_left, cur_val, cur_val - left_val)
                return (next_tree_idx, next_val)
            final_idx, _ = jax.lax.fori_loop(0, n_levels, body_fun, (0, target_val), unroll=True)
            return final_idx
        tree_indices = jax.vmap(_retrieve_single)(target_values)
        indices = tree_indices - 2**n_levels + 1
        return indices, target_values

    def get_root(self):
        return self.state.tree[0]

    def add(self, priority: float):
        """Adds a single data point with its priority."""
        self.state = self._jitted_add_batch(self.state, jnp.array([priority]), self.n_levels, self.capacity, self._jitted_propagate_tree)

    def add_batch(self, priorities: jnp.ndarray):
        """Adds a batch of data points with their priorities."""
        self.state = self._jitted_add_batch(self.state, priorities, self.n_levels, self.capacity, self._jitted_propagate_tree)

    def update(self, priority: float, index: int):
        self.state = self._jitted_update_batch(self.state, jnp.array([priority]), jnp.array([index]), self.n_levels, self._jitted_propagate_tree)

    def update_batch(self, priorities: jnp.ndarray, indices: jnp.ndarray):
        assert priorities.shape[0] == indices.shape[0], "Priorities and indices must have the same batch size"
        self.state = self._jitted_update_batch(self.state, priorities, indices, self.n_levels, self._jitted_propagate_tree)

    def sample(self, rng: PRNGKey, batch_size: int):
        indices, target_values = SumTree._jitted_sample_batch(rng, self.state, batch_size, self.n_levels)
        assert (indices < self.capacity).all(), "Sampled indices must be less than capacity"
        return indices

    def print_tree(self):
        for level in range(self.n_levels+1):
            print(self.state.tree[2**level-1 : 2**(level+1)-1])


class MinTree(SumTree):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.n_levels = int(jnp.ceil(jnp.log2(capacity)))
        self.state = TreeState(
            tree=jnp.full(2**(self.n_levels+1) - 1, jnp.inf),
            pointer=0,
            n_entries=0,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("n_levels"))
    def _jitted_propagate_tree(tree, tree_idx, new_value, n_levels):
        new_tree = tree.at[tree_idx].set(new_value)
        def body_fun(i, carry):
            cur_tree_idx, cur_tree = carry
            parent_idx = (cur_tree_idx - 1) // 2
            left_child, right_child = 2 * parent_idx + 1, 2 * parent_idx + 2
            new_value = jnp.minimum(cur_tree[left_child], cur_tree[right_child])
            return (parent_idx, cur_tree.at[parent_idx].set(new_value))
        return jax.lax.fori_loop(0, n_levels, body_fun, (tree_idx, new_tree), unroll=True)[1]


if __name__ == "__main__":
    import time
    start_time = time.time()
    sample_size = 1000_00
    sumtree = SumTree(capacity=sample_size)
    for i in range(sample_size):
        sumtree.add(i)
        _ = sumtree.sample(jax.random.PRNGKey(1), 256)

    end_time = time.time()
    print(f"Time taken to add {sample_size} elements: {end_time - start_time:.4f} seconds")
