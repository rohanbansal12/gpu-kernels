"""Torch softmax variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch


def softmax_torch(x: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    """Eager Torch softmax implementation."""
    raise NotImplementedError(f"implement softmax_torch with axis={axis}")


def softmax_torch_compile(x: torch.Tensor, *, axis: int = -1) -> torch.Tensor:
    """``torch.compile`` softmax implementation, cached after first use."""
    return _compiled_softmax()(x, axis=axis)


@cache
def _compiled_softmax() -> Callable[..., torch.Tensor]:
    return torch.compile(softmax_torch)
