"""Triton SiLU variant."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any

import torch

from gpu_kernels import runtime

DEFAULT_BLOCK = (1024,)


def silu_triton(
    x: torch.Tensor,
    *,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Apply SiLU to a CUDA tensor with a 1-D Triton program grid."""
    triton = runtime.require_triton()
    if len(block_shape) != 1:
        raise ValueError(f"silu_triton expects a 1-D block_shape, got {block_shape}")
    if not x.is_cuda:
        raise RuntimeError("silu_triton requires a CUDA tensor")

    (block_size,) = block_shape
    x_flat = x.contiguous().view(-1)
    out = torch.empty_like(x_flat)
    n = x_flat.numel()
    grid = (triton.cdiv(n, block_size),)
    _silu_kernel()[grid](x_flat, out, n, BLOCK_SIZE=block_size)
    return out.view_as(x)


@cache
def _silu_kernel() -> Any:
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
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(o_ptr + offsets, x * tl.sigmoid(x), mask=mask)

    return triton.jit(kernel)
