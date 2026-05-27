"""Readable SiLU-multiply oracle."""

from __future__ import annotations

from typing import Any


def silu_mul_naive(x: Any, gate: Any) -> Any:
    """Return ``silu(x) * gate`` as the correctness oracle."""
    raise NotImplementedError("implement silu_mul_naive")
