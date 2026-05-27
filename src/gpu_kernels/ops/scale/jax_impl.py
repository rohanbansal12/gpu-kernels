"""JAX scale variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def scale_jax(x: Any) -> Any:
    """Plain JAX scale implementation."""
    return x * 2


def scale_jax_jit(x: Any) -> Any:
    """Jitted JAX scale implementation, cached after first use."""
    return _jitted_scale()(x)


@cache
def _jitted_scale() -> Any:
    jax = runtime.require_jax()
    return jax.jit(scale_jax)
