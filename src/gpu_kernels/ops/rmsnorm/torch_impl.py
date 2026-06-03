"""Torch RMSNorm variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def rmsnorm_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Eager Torch RMSNorm implementation."""
    raise NotImplementedError(f"implement rmsnorm_torch with eps={eps}")


def rmsnorm_torch_compile(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``torch.compile`` RMSNorm implementation, cached after first use."""
    return _compiled_rmsnorm()(x, weight, eps=eps)


@cache
def _compiled_rmsnorm() -> Callable[..., torch.Tensor]:
    return torch.compile(rmsnorm_torch)
