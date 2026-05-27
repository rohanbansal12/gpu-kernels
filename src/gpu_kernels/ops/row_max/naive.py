"""Readable row-max oracle."""

from __future__ import annotations

from typing import Any


def row_max_naive(x: Any, *, axis: int = -1, keepdims: bool = False) -> Any:
    """Return a row-wise max as the correctness oracle."""
    raise NotImplementedError(f"implement row_max_naive with axis={axis}, keepdims={keepdims}")
