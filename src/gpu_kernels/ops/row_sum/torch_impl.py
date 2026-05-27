"""Torch row-sum variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def row_sum_torch(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """Eager Torch row-sum implementation."""
    raise NotImplementedError(f"implement row_sum_torch with axis={axis}, keepdims={keepdims}")


def row_sum_torch_compile(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """``torch.compile`` row-sum implementation, cached after first use."""
    return _compiled_row_sum()(x, axis=axis, keepdims=keepdims)


@cache
def _compiled_row_sum() -> Any:
    torch = runtime.require_torch()
    return torch.compile(row_sum_torch)
