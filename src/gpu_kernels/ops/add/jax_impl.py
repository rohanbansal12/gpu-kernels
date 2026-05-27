"""JAX add variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def add_jax(x: Any, y: Any) -> Any:
    """Plain JAX add implementation."""
    return x + y


def add_jax_jit(x: Any, y: Any) -> Any:
    """Jitted JAX add implementation, cached after first use."""
    return _jitted_add()(x, y)


@cache
def _jitted_add() -> Any:
    jax = runtime.require_jax()
    return jax.jit(add_jax)
