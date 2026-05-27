"""Readable GELU oracle."""

from __future__ import annotations

from typing import Any, Literal

GeluApprox = Literal["none", "tanh"]


def gelu_naive(x: Any, *, approximate: GeluApprox = "tanh") -> Any:
    """Return GELU activation as the correctness oracle."""
    raise NotImplementedError(f"implement gelu_naive with approximate={approximate}")
