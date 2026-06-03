"""Torch GELU variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch

from gpu_kernels.ops.gelu.naive import GeluApprox


def gelu_torch(x: torch.Tensor, *, approximate: GeluApprox = "tanh") -> torch.Tensor:
    """Eager Torch GELU implementation."""
    raise NotImplementedError(f"implement gelu_torch with approximate={approximate}")


def gelu_torch_compile(x: torch.Tensor, *, approximate: GeluApprox = "tanh") -> torch.Tensor:
    """``torch.compile`` GELU implementation, cached after first use."""
    return _compiled_gelu()(x, approximate=approximate)


@cache
def _compiled_gelu() -> Callable[..., torch.Tensor]:
    return torch.compile(gelu_torch)
