"""Torch bias-add variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def bias_add_torch(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Eager Torch bias-add implementation."""
    return x + bias


def bias_add_torch_compile(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``torch.compile`` bias-add implementation, cached after first use."""
    return _compiled_bias_add()(x, bias)


@cache
def _compiled_bias_add() -> Callable[..., torch.Tensor]:
    return torch.compile(bias_add_torch)
