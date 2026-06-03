"""Torch row-sum variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def row_sum_torch(
    x: torch.Tensor,
    *,
    axis: int = -1,
    keepdims: bool = False,
) -> torch.Tensor:
    """Eager Torch row-sum implementation."""
    raise NotImplementedError(f"implement row_sum_torch with axis={axis}, keepdims={keepdims}")


def row_sum_torch_compile(
    x: torch.Tensor,
    *,
    axis: int = -1,
    keepdims: bool = False,
) -> torch.Tensor:
    """``torch.compile`` row-sum implementation, cached after first use."""
    return _compiled_row_sum()(x, axis=axis, keepdims=keepdims)


@cache
def _compiled_row_sum() -> Callable[..., torch.Tensor]:
    return torch.compile(row_sum_torch)
