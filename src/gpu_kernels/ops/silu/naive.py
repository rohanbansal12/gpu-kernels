"""Readable SiLU oracle."""

from __future__ import annotations

from typing import Any


def silu_naive(x: Any) -> Any:
    """Return SiLU activation as the correctness oracle."""
    return x / (1 + (-x).exp()) if hasattr(x, "exp") else x / (1 + __import__("numpy").exp(-x))
