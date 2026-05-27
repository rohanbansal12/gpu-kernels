"""Readable RMSNorm oracle."""

from __future__ import annotations

from typing import Any


def rmsnorm_naive(x: Any, weight: Any, *, eps: float = 1e-6) -> Any:
    """Return RMSNorm as the correctness oracle."""
    raise NotImplementedError(f"implement rmsnorm_naive with eps={eps}")
