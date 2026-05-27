"""Readable LayerNorm oracle."""

from __future__ import annotations

from typing import Any


def layernorm_naive(x: Any, weight: Any, bias: Any, *, eps: float = 1e-5) -> Any:
    """Return LayerNorm as the correctness oracle."""
    raise NotImplementedError(f"implement layernorm_naive with eps={eps}")
