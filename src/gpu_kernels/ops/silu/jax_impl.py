"""JAX SiLU variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def silu_jax(x: Any) -> Any:
    """Plain JAX SiLU implementation."""
    jax = runtime.require_jax()
    return jax.nn.silu(x)


def silu_jax_jit(x: Any) -> Any:
    """Jitted JAX SiLU implementation, cached after first use."""
    return _jitted_silu()(x)


@cache
def _jitted_silu() -> Any:
    jax = runtime.require_jax()
    return jax.jit(silu_jax)
