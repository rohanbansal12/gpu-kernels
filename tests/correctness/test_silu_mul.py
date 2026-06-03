"""Correctness tests for silu_mul variants."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from gpu_kernels import runtime
from gpu_kernels.ops.silu_mul import (
    silu_mul_jax_jit,
    silu_mul_naive,
    silu_mul_torch,
    silu_mul_torch_compile,
    silu_mul_triton,
)


def test_silu_mul_naive_numpy() -> None:
    x = np.array([1.0, -2.0, 0.0, 3.5], dtype=np.float32)
    gate = np.array([0.5, -1.0, 2.0, 3.0], dtype=np.float32)
    expected = (x / (1 + np.exp(-x))) * gate
    np.testing.assert_allclose(silu_mul_naive(x, gate), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(32,), (7, 13)])
@pytest.mark.torch
def test_silu_mul_torch_matches_naive(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape, dtype=torch.float32)
    gate = torch.randn(shape, dtype=torch.float32)
    torch.testing.assert_close(silu_mul_torch(x, gate), silu_mul_naive(x, gate))


@pytest.mark.parametrize("shape", [(32,), (7, 13)])
@pytest.mark.torch
def test_silu_mul_torch_compile_matches_naive(shape: tuple[int, ...]) -> None:
    x = torch.randn(shape, dtype=torch.float32)
    gate = torch.randn(shape, dtype=torch.float32)
    torch.testing.assert_close(silu_mul_torch_compile(x, gate), silu_mul_naive(x, gate))


@pytest.mark.torch
def test_silu_mul_torch_handles_non_contiguous_inputs() -> None:
    x = torch.randn((8, 16), dtype=torch.float32).t()
    gate = torch.randn((8, 16), dtype=torch.float32).t()
    assert not x.is_contiguous()
    assert not gate.is_contiguous()
    torch.testing.assert_close(silu_mul_torch(x, gate), silu_mul_naive(x, gate))


@pytest.mark.jax
def test_silu_mul_jax_jit_matches_naive(require_jax: Any) -> None:
    _ = require_jax
    x = runtime.jax_random((7, 13), dtype="fp32", seed=0)
    gate = runtime.jax_random((7, 13), dtype="fp32", seed=1)
    expected = silu_mul_naive(runtime.jax_to_numpy(x), runtime.jax_to_numpy(gate))
    np.testing.assert_allclose(
        runtime.jax_to_numpy(silu_mul_jax_jit(x, gate)),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.triton
@pytest.mark.gpu
def test_silu_mul_triton_matches_naive_1d_odd_size(
    require_cuda: Any,
    require_triton: Any,
) -> None:
    torch = require_cuda
    _ = require_triton
    x = torch.randn((33,), device="cuda", dtype=torch.float32)
    gate = torch.randn((33,), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        silu_mul_triton(x, gate, block_shape=(16,)),
        silu_mul_naive(x, gate),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.triton
@pytest.mark.gpu
def test_silu_mul_triton_matches_naive_2d(
    require_cuda: Any,
    require_triton: Any,
) -> None:
    torch = require_cuda
    _ = require_triton
    x = torch.randn((5, 17), device="cuda", dtype=torch.float32)
    gate = torch.randn((5, 17), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        silu_mul_triton(x, gate, block_shape=(32,)),
        silu_mul_naive(x, gate),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.triton
@pytest.mark.gpu
def test_silu_mul_triton_matches_naive_bf16(
    require_cuda: Any,
    require_triton: Any,
) -> None:
    torch = require_cuda
    _ = require_triton
    x = torch.randn((65,), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((65,), device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        silu_mul_triton(x, gate, block_shape=(32,)),
        silu_mul_naive(x, gate),
        rtol=5e-2,
        atol=5e-2,
    )
