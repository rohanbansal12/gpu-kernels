"""JAX softmax variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def softmax_jax(x: Any, *, axis: int = -1) -> Any:
    """Plain JAX softmax implementation."""
    raise NotImplementedError(f"implement softmax_jax with axis={axis}")


def softmax_jax_jit(x: Any, *, axis: int = -1) -> Any:
    """Jitted JAX softmax implementation, cached after first use."""
    return _jitted_softmax()(x, axis=axis)


@cache
def _jitted_softmax() -> Any:
    jax = runtime.require_jax()
    return jax.jit(softmax_jax, static_argnames=("axis",))
