"""Triton row-sum variant template."""

from __future__ import annotations

from typing import Any

DEFAULT_BLOCK = (1, 1024)


def row_sum_triton(
    x: Any,
    *,
    axis: int = -1,
    keepdims: bool = False,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> Any:
    """Hand-written Triton row-sum kernel."""
    raise NotImplementedError(
        f"implement row_sum_triton with axis={axis}, keepdims={keepdims}, block_shape={block_shape}"
    )
