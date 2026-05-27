"""JAX row-sum variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def row_sum_jax(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """Plain JAX row-sum implementation."""
    raise NotImplementedError(f"implement row_sum_jax with axis={axis}, keepdims={keepdims}")


def row_sum_jax_jit(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """Jitted JAX row-sum implementation, cached after first use."""
    return _jitted_row_sum()(x, axis=axis, keepdims=keepdims)


@cache
def _jitted_row_sum() -> Any:
    jax = runtime.require_jax()
    return jax.jit(row_sum_jax, static_argnames=("axis", "keepdims"))
