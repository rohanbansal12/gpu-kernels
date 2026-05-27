"""Torch softmax variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def softmax_torch(x: Any, *, axis: int = -1) -> Any:
    """Eager Torch softmax implementation."""
    raise NotImplementedError(f"implement softmax_torch with axis={axis}")


def softmax_torch_compile(x: Any, *, axis: int = -1) -> Any:
    """``torch.compile`` softmax implementation, cached after first use."""
    return _compiled_softmax()(x, axis=axis)


@cache
def _compiled_softmax() -> Any:
    torch = runtime.require_torch()
    return torch.compile(softmax_torch)
