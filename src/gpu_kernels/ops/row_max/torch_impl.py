"""Torch row-max variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def row_max_torch(
    x: torch.Tensor,
    *,
    axis: int = -1,
    keepdims: bool = False,
) -> torch.Tensor:
    """Eager Torch row-max implementation."""
    raise NotImplementedError(f"implement row_max_torch with axis={axis}, keepdims={keepdims}")


def row_max_torch_compile(
    x: torch.Tensor,
    *,
    axis: int = -1,
    keepdims: bool = False,
) -> torch.Tensor:
    """``torch.compile`` row-max implementation, cached after first use."""
    return _compiled_row_max()(x, axis=axis, keepdims=keepdims)


@cache
def _compiled_row_max() -> Callable[..., torch.Tensor]:
    return torch.compile(row_max_torch)
