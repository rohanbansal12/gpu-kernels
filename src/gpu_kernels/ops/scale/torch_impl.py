"""Torch scale variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def scale_torch(x: torch.Tensor) -> torch.Tensor:
    """Eager Torch scale implementation."""
    return x * 2


def scale_torch_compile(x: torch.Tensor) -> torch.Tensor:
    """``torch.compile`` scale implementation, cached after first use."""
    return _compiled_scale()(x)


@cache
def _compiled_scale() -> Callable[..., torch.Tensor]:
    return torch.compile(scale_torch)
