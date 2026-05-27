"""Torch LayerNorm variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def layernorm_torch(x: Any, weight: Any, bias: Any, *, eps: float = 1e-5) -> Any:
    """Eager Torch LayerNorm implementation."""
    raise NotImplementedError(f"implement layernorm_torch with eps={eps}")


def layernorm_torch_compile(x: Any, weight: Any, bias: Any, *, eps: float = 1e-5) -> Any:
    """``torch.compile`` LayerNorm implementation, cached after first use."""
    return _compiled_layernorm()(x, weight, bias, eps=eps)


@cache
def _compiled_layernorm() -> Any:
    torch = runtime.require_torch()
    return torch.compile(layernorm_torch)
