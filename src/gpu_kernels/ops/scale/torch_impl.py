"""Torch scale variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def scale_torch(x: Any) -> Any:
    """Eager Torch scale implementation."""
    return x * 2


def scale_torch_compile(x: Any) -> Any:
    """``torch.compile`` scale implementation, cached after first use."""
    return _compiled_scale()(x)


@cache
def _compiled_scale() -> Any:
    torch = runtime.require_torch()
    return torch.compile(scale_torch)
