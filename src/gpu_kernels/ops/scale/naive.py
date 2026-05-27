"""Readable scale oracle."""

from __future__ import annotations

from typing import Any


def scale_naive(x: Any) -> Any:
    """Return ``x * 2`` as the correctness oracle.

    This intentionally relies on the input framework's normal multiplication
    semantics so the same oracle works for Torch tensors, JAX arrays, and
    NumPy arrays during early development.
    """
    return x * 2
