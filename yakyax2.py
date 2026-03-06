# v2: The Graph-Fluent Interface

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from typing import Callable, Union, Tuple, Any

class YakTensor:
    """
    A Graph-based JAX wrapper. 
    It views the tensor as a 'signal' moving through a pipeline.
    """
    def __init__(self, data: Any, node_fn: Callable = lambda x: x, name: str = "root"):
        self.data = data  # This can be a JAX array or another YakTensor node
        self.node_fn = node_fn
        self.name = name

    def __call__(self, x: Any) -> Any:
        """Allows the YakTensor object to act as a standard JAX function."""
        return self.node_fn(x)

    def _chain(self, next_fn: Callable, name: str) -> 'YakTensor':
        """The core composition engine."""
        # We wrap the current node_fn with the next_fn
        composed = lambda x: next_fn(self.node_fn(x))
        return YakTensor(self.data, composed, f"{self.name}->{name}")

    # --- Fluent API ---
    def pipe(self, fn: Callable) -> 'YakTensor':
        """Inject any JAX/Custom function into the chain."""
        return self._chain(fn, fn.__name__ if hasattr(fn, '__name__') else "pipe")

    def map(self, axis: int = 0) -> 'YakTensor':
        """Vectorize the current pipeline logic."""
        return self._chain(lambda x: vmap(lambda inner: inner)(x), "vmap")

    def fuse(self) -> 'YakTensor':
        """Immediately JIT the accumulated pipeline logic."""
        compiled = jit(self.node_fn)
        return YakTensor(self.data, compiled, f"jit({self.name})")

    # --- Math Overloads (Implicitly lazy) ---
    def __add__(self, other):
        if isinstance(other, YakTensor):
            # This is where the magic happens: Branching
            # We create a new node that evaluates both branches
            fn = lambda x: self.node_fn(x) + other.node_fn(x)
            return YakTensor(self.data, fn, "branch_add")
        return self._chain(lambda x: x + other, "add")

    def __mul__(self, other):
        return self._chain(lambda x: x * other, "mul")

    def __matmul__(self, other):
        return self._chain(lambda x: x @ other, "matmul")

    # --- Terminal Ops ---
    def value(self) -> Any:
        """The 'Moment of Reality': Materialize the graph."""
        return self.node_fn(self.data)

    def __repr__(self):
        return f"YakTensor[ {self.name} ]"

# YAKYAX alias
yakyax = YakTensor
