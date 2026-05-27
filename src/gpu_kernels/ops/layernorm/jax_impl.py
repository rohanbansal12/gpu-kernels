"""JAX LayerNorm variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def layernorm_jax(x: Any, weight: Any, bias: Any, *, eps: float = 1e-5) -> Any:
    """Plain JAX LayerNorm implementation."""
    raise NotImplementedError(f"implement layernorm_jax with eps={eps}")


def layernorm_jax_jit(x: Any, weight: Any, bias: Any, *, eps: float = 1e-5) -> Any:
    """Jitted JAX LayerNorm implementation, cached after first use."""
    return _jitted_layernorm()(x, weight, bias, eps=eps)


@cache
def _jitted_layernorm() -> Any:
    jax = runtime.require_jax()
    return jax.jit(layernorm_jax, static_argnames=("eps",))
