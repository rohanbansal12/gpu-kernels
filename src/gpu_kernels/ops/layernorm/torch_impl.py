"""Torch LayerNorm variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def layernorm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Eager Torch LayerNorm implementation."""
    raise NotImplementedError(f"implement layernorm_torch with eps={eps}")


def layernorm_torch_compile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``torch.compile`` LayerNorm implementation, cached after first use."""
    return _compiled_layernorm()(x, weight, bias, eps=eps)


@cache
def _compiled_layernorm() -> Callable[..., torch.Tensor]:
    return torch.compile(layernorm_torch)
