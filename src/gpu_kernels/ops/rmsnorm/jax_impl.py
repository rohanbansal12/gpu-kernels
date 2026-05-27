"""JAX RMSNorm variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def rmsnorm_jax(x: Any, weight: Any, *, eps: float = 1e-6) -> Any:
    """Plain JAX RMSNorm implementation."""
    raise NotImplementedError(f"implement rmsnorm_jax with eps={eps}")


def rmsnorm_jax_jit(x: Any, weight: Any, *, eps: float = 1e-6) -> Any:
    """Jitted JAX RMSNorm implementation, cached after first use."""
    return _jitted_rmsnorm()(x, weight, eps=eps)


@cache
def _jitted_rmsnorm() -> Any:
    jax = runtime.require_jax()
    return jax.jit(rmsnorm_jax, static_argnames=("eps",))
