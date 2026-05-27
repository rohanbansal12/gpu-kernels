"""Triton SiLU-multiply variant template."""

from __future__ import annotations

from typing import Any

DEFAULT_BLOCK = (1024,)


def silu_mul_triton(
    x: Any,
    gate: Any,
    *,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> Any:
    """Hand-written Triton SiLU-multiply kernel."""
    raise NotImplementedError(f"implement silu_mul_triton with block_shape={block_shape}")
