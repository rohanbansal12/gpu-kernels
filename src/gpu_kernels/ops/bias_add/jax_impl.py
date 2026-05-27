"""JAX bias-add variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def bias_add_jax(x: Any, bias: Any) -> Any:
    """Plain JAX bias-add implementation."""
    return x + bias


def bias_add_jax_jit(x: Any, bias: Any) -> Any:
    """Jitted JAX bias-add implementation, cached after first use."""
    return _jitted_bias_add()(x, bias)


@cache
def _jitted_bias_add() -> Any:
    jax = runtime.require_jax()
    return jax.jit(bias_add_jax)
