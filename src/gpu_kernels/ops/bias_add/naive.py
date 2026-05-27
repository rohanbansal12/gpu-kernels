"""Readable bias-add oracle."""

from __future__ import annotations

from typing import Any


def bias_add_naive(x: Any, bias: Any) -> Any:
    """Return ``x + bias`` as the correctness oracle."""
    return x + bias
