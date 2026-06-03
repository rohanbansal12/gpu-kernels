"""Triton row-max variant template."""

from __future__ import annotations

import torch

DEFAULT_BLOCK = (1, 1024)


def row_max_triton(
    x: torch.Tensor,
    *,
    axis: int = -1,
    keepdims: bool = False,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Hand-written Triton row-max kernel."""
    raise NotImplementedError(
        f"implement row_max_triton with axis={axis}, keepdims={keepdims}, block_shape={block_shape}"
    )
