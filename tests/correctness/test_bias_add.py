from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from gpu_kernels import runtime
from gpu_kernels.ops.bias_add import (
    bias_add_jax_jit,
    bias_add_naive,
    bias_add_torch,
    bias_add_torch_compile,
    bias_add_triton,
)


def test_bias_add_naive_numpy() -> None:
    x = np.array([[1.0, -2.0, 3.5], [4.0, 5.0, -1.5]], dtype=np.float32)
    bias = np.array([0.5, 1.0, -2.0], dtype=np.float32)
    expected = np.array([[1.5, -1.0, 1.5], [4.5, 6.0, -3.5]], dtype=np.float32)
    np.testing.assert_allclose(bias_add_naive(x, bias), expected)


@pytest.mark.torch
def test_bias_add_torch_matches_naive() -> None:
    x = torch.randn((8, 17), dtype=torch.float32)
    bias = torch.randn((17,), dtype=torch.float32)
    torch.testing.assert_close(bias_add_torch(x, bias), bias_add_naive(x, bias))


@pytest.mark.torch
def test_bias_add_torch_compile_matches_naive() -> None:
    x = torch.randn((8, 17), dtype=torch.float32)
    bias = torch.randn((17,), dtype=torch.float32)
    torch.testing.assert_close(bias_add_torch_compile(x, bias), bias_add_naive(x, bias))


@pytest.mark.jax
def test_bias_add_jax_jit_matches_naive(require_jax: Any) -> None:
    _ = require_jax
    x = runtime.jax_random((8, 17), dtype="fp32", seed=0)
    bias = runtime.jax_random((17,), dtype="fp32", seed=1)
    np.testing.assert_allclose(
        runtime.jax_to_numpy(bias_add_jax_jit(x, bias)),
        runtime.jax_to_numpy(bias_add_naive(x, bias)),
    )


@pytest.mark.triton
@pytest.mark.gpu
def test_bias_add_triton_matches_naive(require_cuda: Any, require_triton: Any) -> None:
    torch = require_cuda
    _ = require_triton
    x = torch.randn((5, 17), device="cuda", dtype=torch.float32)
    bias = torch.randn((17,), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        bias_add_triton(x, bias, block_shape=(2, 8)),
        bias_add_naive(x, bias),
    )
