"""Readable softmax oracle."""

from __future__ import annotations

from typing import Any


def softmax_naive(x: Any, *, axis: int = -1) -> Any:
    """Return stable softmax as the correctness oracle."""
    raise NotImplementedError(f"implement softmax_naive with axis={axis}")
