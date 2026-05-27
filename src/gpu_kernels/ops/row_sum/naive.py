"""Readable row-sum oracle."""

from __future__ import annotations

from typing import Any


def row_sum_naive(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """Return a row-wise sum as the correctness oracle."""
    raise NotImplementedError(f"implement row_sum_naive with axis={axis}, keepdims={keepdims}")
