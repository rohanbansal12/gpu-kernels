"""Triton LayerNorm variant template."""

from __future__ import annotations

from typing import Any

DEFAULT_BLOCK = (1, 1024)


def layernorm_triton(
    x: Any,
    weight: Any,
    bias: Any,
    *,
    eps: float = 1e-5,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> Any:
    """Hand-written Triton LayerNorm kernel."""
    raise NotImplementedError(
        f"implement layernorm_triton with eps={eps}, block_shape={block_shape}"
    )
