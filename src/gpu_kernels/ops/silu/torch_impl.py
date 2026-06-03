"""Torch SiLU variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def silu_torch(x: torch.Tensor) -> torch.Tensor:
    """Eager Torch SiLU implementation."""
    return torch.nn.functional.silu(x)


def silu_torch_compile(x: torch.Tensor) -> torch.Tensor:
    """``torch.compile`` SiLU implementation, cached after first use."""
    return _compiled_silu()(x)


@cache
def _compiled_silu() -> Callable[..., torch.Tensor]:
    return torch.compile(silu_torch)
