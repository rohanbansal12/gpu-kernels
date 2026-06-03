"""Triton RMSNorm variant template."""

from __future__ import annotations

import torch

DEFAULT_BLOCK = (1, 1024)


def rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float = 1e-6,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Hand-written Triton RMSNorm kernel."""
    raise NotImplementedError(f"implement rmsnorm_triton with eps={eps}, block_shape={block_shape}")
