"""Triton SiLU-multiply variant template."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any

import torch

from gpu_kernels import runtime

DEFAULT_BLOCK = (1024,)


def silu_mul_triton(
    x: torch.Tensor,
    gate: torch.Tensor,
    *,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> torch.Tensor:
    """Hand-written Triton SiLU-multiply kernel."""
    triton = runtime.require_triton()
    if len(block_shape) != 1:
        raise ValueError(f"silu_triton expects a 1-D block_shape, got {block_shape}")
    if not x.is_cuda or not gate.is_cuda:
        raise RuntimeError("silu_triton requires a CUDA tensor")
    
    (block_size,) = block_shape
    x_flat = x.contiguous().view(-1)
    gate_flat = gate.contiguous().view(-1)
    out = torch.empty_like(x_flat)
    n = x_flat.numel()
    grid = (triton.cdiv(n, block_size),)
    _silu_mul_kernel()[grid](x_flat, gate_flat, out, n, BLOCK_SIZE=block_size)
    return out.view_as(x)


@cache
def _silu_mul_kernel() -> Any:
    triton = runtime.require_triton()
    tl = import_module("triton.language")

    def kernel(
        x_ptr,
        gate_ptr,
        o_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,  # noqa: N803  # pyright: ignore[reportInvalidTypeForm]
    ) -> None:
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
        tl.store(o_ptr + offsets, gate * x * tl.sigmoid(x), mask=mask)

    return triton.jit(kernel)
