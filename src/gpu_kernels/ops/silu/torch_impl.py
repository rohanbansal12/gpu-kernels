"""Torch SiLU variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def silu_torch(x: Any) -> Any:
    """Eager Torch SiLU implementation."""
    torch = runtime.require_torch()
    return torch.nn.functional.silu(x)


def silu_torch_compile(x: Any) -> Any:
    """``torch.compile`` SiLU implementation, cached after first use."""
    return _compiled_silu()(x)


@cache
def _compiled_silu() -> Any:
    torch = runtime.require_torch()
    return torch.compile(silu_torch)
