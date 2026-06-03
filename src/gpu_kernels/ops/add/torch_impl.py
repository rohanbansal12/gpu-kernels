"""Torch add variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def add_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Eager Torch add implementation."""
    return x + y


def add_torch_compile(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``torch.compile`` add implementation, cached after first use."""
    return _compiled_add()(x, y)


@cache
def _compiled_add() -> Callable[..., torch.Tensor]:
    return torch.compile(add_torch)
