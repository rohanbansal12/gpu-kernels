"""Triton LayerNorm variant template."""

from __future__ import annotations

import torch

DEFAULT_BLOCK = (1, 1024)


def layernorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Hand-written Triton LayerNorm kernel."""
    raise NotImplementedError(
        f"implement layernorm_triton with eps={eps}, block_shape={block_shape}"
    )
