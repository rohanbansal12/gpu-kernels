"""Triton bias-add variant."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any

from gpu_kernels import runtime

DEFAULT_BLOCK = (16, 1024)


def bias_add_triton(
    x: Any,
    bias: Any,
    *,
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> Any:
    """Add a 1-D bias vector to each row of a 2-D CUDA tensor."""
    torch = runtime.require_torch()
    triton = runtime.require_triton()
    if len(block_shape) != 2:
        raise ValueError(f"bias_add_triton expects a 2-D block_shape, got {block_shape}")
    if not isinstance(x, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("bias_add_triton expects Torch tensors")
    if not x.is_cuda or not bias.is_cuda:
        raise RuntimeError("bias_add_triton requires CUDA tensors")
    if x.ndim != 2:
        raise ValueError(f"bias_add_triton expects 2-D x, got shape {tuple(x.shape)}")
    if bias.ndim != 1:
        raise ValueError(f"bias_add_triton expects 1-D bias, got shape {tuple(bias.shape)}")

    rows, hidden = x.shape
    if bias.shape[0] != hidden:
        raise ValueError(f"bias shape {tuple(bias.shape)} does not match hidden dim {hidden}")
    if x.dtype != bias.dtype:
        raise ValueError(f"bias_add_triton expects matching dtypes, got {x.dtype} and {bias.dtype}")

    block_m, block_n = block_shape
    x_contig = x.contiguous()
    bias_contig = bias.contiguous()
    out = torch.empty_like(x_contig)
    grid = (triton.cdiv(rows, block_m), triton.cdiv(hidden, block_n))
    _bias_add_kernel()[grid](
        x_contig,
        bias_contig,
        out,
        rows,
        hidden,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return out.view_as(x)


@cache
def _bias_add_kernel() -> Any:
    triton = runtime.require_triton()
    tl = import_module("triton.language")

    def kernel(
        x_ptr,
        bias_ptr,
        o_ptr,
        rows,
        hidden,
        # Triton meta-parameters are conventionally uppercase.
        BLOCK_M: tl.constexpr,  # noqa: N803  # pyright: ignore[reportInvalidTypeForm]
        BLOCK_N: tl.constexpr,  # noqa: N803  # pyright: ignore[reportInvalidTypeForm]
    ) -> None:
        row_pid = tl.program_id(0)
        col_pid = tl.program_id(1)
        row_offsets = row_pid * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = col_pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets = row_offsets[:, None] * hidden + col_offsets[None, :]
        mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < hidden)
        x = tl.load(x_ptr + offsets, mask=mask)
        bias = tl.load(bias_ptr + col_offsets, mask=col_offsets < hidden)
        tl.store(o_ptr + offsets, x + bias[None, :], mask=mask)

    return triton.jit(kernel)
