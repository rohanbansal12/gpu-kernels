"""Readable SiLU-multiply oracle."""

from __future__ import annotations

from typing import Any


def silu_mul_naive(x: Any, gate: Any) -> Any:
    """Return ``silu(x) * gate`` as the correctness oracle."""
    silu_x = x / (1 + (-x).exp()) if hasattr(x, "exp") else x / (1 + __import__("numpy").exp(-x))
    return silu_x * gate