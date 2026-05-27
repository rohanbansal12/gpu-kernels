"""Torch SiLU-multiply variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def silu_mul_torch(x: Any, gate: Any) -> Any:
    """Eager Torch SiLU-multiply implementation."""
    raise NotImplementedError("implement silu_mul_torch")


def silu_mul_torch_compile(x: Any, gate: Any) -> Any:
    """``torch.compile`` SiLU-multiply implementation, cached after first use."""
    return _compiled_silu_mul()(x, gate)


@cache
def _compiled_silu_mul() -> Any:
    torch = runtime.require_torch()
    return torch.compile(silu_mul_torch)
