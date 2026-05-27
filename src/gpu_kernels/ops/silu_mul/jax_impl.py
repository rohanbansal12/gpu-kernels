"""JAX SiLU-multiply variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def silu_mul_jax(x: Any, gate: Any) -> Any:
    """Plain JAX SiLU-multiply implementation."""
    raise NotImplementedError("implement silu_mul_jax")


def silu_mul_jax_jit(x: Any, gate: Any) -> Any:
    """Jitted JAX SiLU-multiply implementation, cached after first use."""
    return _jitted_silu_mul()(x, gate)


@cache
def _jitted_silu_mul() -> Any:
    jax = runtime.require_jax()
    return jax.jit(silu_mul_jax)
