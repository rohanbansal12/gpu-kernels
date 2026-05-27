"""Torch add variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def add_torch(x: Any, y: Any) -> Any:
    """Eager Torch add implementation."""
    return x + y


def add_torch_compile(x: Any, y: Any) -> Any:
    """``torch.compile`` add implementation, cached after first use."""
    return _compiled_add()(x, y)


@cache
def _compiled_add() -> Any:
    torch = runtime.require_torch()
    return torch.compile(add_torch)
