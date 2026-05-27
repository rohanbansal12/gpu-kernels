"""Triton GELU variant template."""

from __future__ import annotations

from typing import Any

from gpu_kernels.ops.gelu.naive import GeluApprox

DEFAULT_BLOCK = (1024,)


def gelu_triton(
    x: Any,
    *,
    approximate: GeluApprox = "tanh",
    block_shape: tuple[int, ...] = DEFAULT_BLOCK,
) -> Any:
    """Hand-written Triton GELU kernel."""
    raise NotImplementedError(
        f"implement gelu_triton with approximate={approximate}, block_shape={block_shape}"
    )
