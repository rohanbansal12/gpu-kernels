"""Torch bias-add variants."""

from __future__ import annotations

from functools import cache
from typing import Any

from gpu_kernels import runtime


def bias_add_torch(x: Any, bias: Any) -> Any:
    """Eager Torch bias-add implementation."""
    return x + bias


def bias_add_torch_compile(x: Any, bias: Any) -> Any:
    """``torch.compile`` bias-add implementation, cached after first use."""
    return _compiled_bias_add()(x, bias)


@cache
def _compiled_bias_add() -> Any:
    torch = runtime.require_torch()
    return torch.compile(bias_add_torch)
