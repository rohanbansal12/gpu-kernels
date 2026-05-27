"""JAX GELU variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime
from gpu_kernels.ops.gelu.naive import GeluApprox


def gelu_jax(x: Any, *, approximate: GeluApprox = "tanh") -> Any:
    """Plain JAX GELU implementation."""
    raise NotImplementedError(f"implement gelu_jax with approximate={approximate}")


def gelu_jax_jit(x: Any, *, approximate: GeluApprox = "tanh") -> Any:
    """Jitted JAX GELU implementation, cached after first use."""
    return _jitted_gelu()(x, approximate=approximate)


@cache
def _jitted_gelu() -> Any:
    jax = runtime.require_jax()
    return jax.jit(gelu_jax, static_argnames=("approximate",))
