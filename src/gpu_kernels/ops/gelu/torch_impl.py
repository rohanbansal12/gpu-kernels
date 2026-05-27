"""Torch GELU variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime
from gpu_kernels.ops.gelu.naive import GeluApprox


def gelu_torch(x: Any, *, approximate: GeluApprox = "tanh") -> Any:
    """Eager Torch GELU implementation."""
    raise NotImplementedError(f"implement gelu_torch with approximate={approximate}")


def gelu_torch_compile(x: Any, *, approximate: GeluApprox = "tanh") -> Any:
    """``torch.compile`` GELU implementation, cached after first use."""
    return _compiled_gelu()(x, approximate=approximate)


@cache
def _compiled_gelu() -> Any:
    torch = runtime.require_torch()
    return torch.compile(gelu_torch)
