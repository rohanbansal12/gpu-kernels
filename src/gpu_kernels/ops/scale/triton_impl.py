"""Triton scale variant."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any

import torch

from gpu_kernels import runtime

DEFAULT_BLOCK = (1024,)


def scale_triton(
    x: torch.Tensor,
    *,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Scale a CUDA tensor by two with a 1-D Triton program grid."""
    triton = runtime.require_triton()
    if len(block_shape) != 1:
        raise ValueError(f"scale_triton expects a 1-D block_shape, got {block_shape}")
    if not isinstance(x, torch.Tensor):
        raise TypeError("scale_triton expects a Torch tensor")
    if not x.is_cuda:
        raise RuntimeError("scale_triton requires a CUDA tensor")

    (block_size,) = block_shape
    x_flat = x.contiguous().view(-1)
    out = torch.empty_like(x_flat)
    n = x_flat.numel()
    grid = (triton.cdiv(n, block_size),)
    _scale_kernel()[grid](x_flat, out, n, BLOCK_SIZE=block_size)
    return out.view_as(x)


@cache
def _scale_kernel() -> Any:
    triton = runtime.require_triton()
    tl = import_module("triton.language")

    def kernel(
        x_ptr,
        o_ptr,
        n_elements,
        # Triton meta-parameters are conventionally uppercase.
        BLOCK_SIZE: tl.constexpr,  # noqa: N803  # pyright: ignore[reportInvalidTypeForm]
    ) -> None:
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(o_ptr + offsets, x * 2, mask=mask)

    return triton.jit(kernel)
