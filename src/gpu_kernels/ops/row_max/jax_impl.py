"""JAX row-max variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def row_max_jax(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """Plain JAX row-max implementation."""
    raise NotImplementedError(f"implement row_max_jax with axis={axis}, keepdims={keepdims}")


def row_max_jax_jit(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """Jitted JAX row-max implementation, cached after first use."""
    return _jitted_row_max()(x, axis=axis, keepdims=keepdims)


@cache
def _jitted_row_max() -> Any:
    jax = runtime.require_jax()
    return jax.jit(row_max_jax, static_argnames=("axis", "keepdims"))
