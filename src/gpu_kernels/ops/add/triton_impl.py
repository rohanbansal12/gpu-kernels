"""Triton add variant."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any

import torch

from gpu_kernels import runtime

DEFAULT_BLOCK = (1024,)


def add_triton(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Add two same-shaped CUDA tensors with a 1-D Triton program grid."""
    triton = runtime.require_triton()
    if len(block_shape) != 1:
        raise ValueError(f"add_triton expects a 1-D block_shape, got {block_shape}")
    if not x.is_cuda or not y.is_cuda:
        raise RuntimeError("add_triton requires CUDA tensors")
    if x.shape != y.shape:
        raise ValueError(f"add_triton expects same-shaped tensors, got {x.shape} and {y.shape}")
    if x.dtype != y.dtype:
        raise ValueError(f"add_triton expects matching dtypes, got {x.dtype} and {y.dtype}")

    (block_size,) = block_shape
    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    out = torch.empty_like(x_flat)
    n = x_flat.numel()
    grid = (triton.cdiv(n, block_size),)
    _add_kernel()[grid](x_flat, y_flat, out, n, BLOCK_SIZE=block_size)
    return out.view_as(x)


@cache
def _add_kernel() -> Any:
    triton = runtime.require_triton()
    tl = import_module("triton.language")

    def kernel(
        x_ptr,
        y_ptr,
        o_ptr,
        n_elements,
        # Triton meta-parameters are conventionally uppercase.
        BLOCK_SIZE: tl.constexpr,  # noqa: N803  # pyright: ignore[reportInvalidTypeForm]
    ) -> None:
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(o_ptr + offsets, x + y, mask=mask)

    return triton.jit(kernel)
