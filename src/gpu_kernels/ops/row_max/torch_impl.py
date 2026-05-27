"""Torch row-max variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def row_max_torch(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """Eager Torch row-max implementation."""
    raise NotImplementedError(f"implement row_max_torch with axis={axis}, keepdims={keepdims}")


def row_max_torch_compile(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """``torch.compile`` row-max implementation, cached after first use."""
    return _compiled_row_max()(x, axis=axis, keepdims=keepdims)


@cache
def _compiled_row_max() -> Any:
    torch = runtime.require_torch()
    return torch.compile(row_max_torch)
