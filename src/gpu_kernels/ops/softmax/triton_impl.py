"""Triton softmax variant template."""

from __future__ import annotations

import torch

DEFAULT_BLOCK = (1, 1024)


def softmax_triton(
    x: torch.Tensor,
    *,
    axis: int = -1,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Hand-written Triton softmax kernel."""
    raise NotImplementedError(
        f"implement softmax_triton with axis={axis}, block_shape={block_shape}"
    )
