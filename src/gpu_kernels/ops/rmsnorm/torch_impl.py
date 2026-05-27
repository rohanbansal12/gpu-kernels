"""Torch RMSNorm variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def rmsnorm_torch(x: Any, weight: Any, *, eps: float = 1e-6) -> Any:
    """Eager Torch RMSNorm implementation."""
    raise NotImplementedError(f"implement rmsnorm_torch with eps={eps}")


def rmsnorm_torch_compile(x: Any, weight: Any, *, eps: float = 1e-6) -> Any:
    """``torch.compile`` RMSNorm implementation, cached after first use."""
    return _compiled_rmsnorm()(x, weight, eps=eps)


@cache
def _compiled_rmsnorm() -> Any:
    torch = runtime.require_torch()
    return torch.compile(rmsnorm_torch)
