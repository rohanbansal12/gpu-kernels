from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from gpu_kernels import runtime
from gpu_kernels.ops.scale import (
    scale_jax_jit,
    scale_naive,
    scale_torch,
    scale_torch_compile,
    scale_triton,
)


def test_scale_naive_numpy() -> None:
    x = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    np.testing.assert_allclose(scale_naive(x), np.array([2.0, -4.0, 7.0], dtype=np.float32))


@pytest.mark.torch
def test_scale_torch_matches_naive() -> None:
    x = torch.randn((32,), dtype=torch.float32)
    torch.testing.assert_close(scale_torch(x), scale_naive(x))


@pytest.mark.torch
def test_scale_torch_compile_matches_naive() -> None:
    x = torch.randn((32,), dtype=torch.float32)
    torch.testing.assert_close(scale_torch_compile(x), scale_naive(x))


@pytest.mark.jax
def test_scale_jax_jit_matches_naive(require_jax: Any) -> None:
    _ = require_jax
    x = runtime.jax_random((32,), dtype="fp32", seed=0)
    np.testing.assert_allclose(
        runtime.jax_to_numpy(scale_jax_jit(x)),
        runtime.jax_to_numpy(scale_naive(x)),
    )


@pytest.mark.triton
@pytest.mark.gpu
def test_scale_triton_matches_naive(require_cuda: Any, require_triton: Any) -> None:
    torch = require_cuda
    _ = require_triton
    x = torch.randn((33,), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(scale_triton(x, block_shape=(16,)), scale_naive(x))
