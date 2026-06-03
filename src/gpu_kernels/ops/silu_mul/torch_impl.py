"""Torch SiLU-multiply variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def silu_mul_torch(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Eager Torch SiLU-multiply implementation."""
    return torch.nn.functional.silu(x) * gate


def silu_mul_torch_compile(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``torch.compile`` SiLU-multiply implementation, cached after first use."""
    return _compiled_silu_mul()(x, gate)


@cache
def _compiled_silu_mul() -> Callable[..., torch.Tensor]:
    return torch.compile(silu_mul_torch)
