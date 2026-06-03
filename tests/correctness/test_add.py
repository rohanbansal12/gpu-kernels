from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from gpu_kernels import runtime
from gpu_kernels.ops.add import add_jax_jit, add_naive, add_torch, add_torch_compile, add_triton


def test_add_naive_numpy() -> None:
    x = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    y = np.array([4.0, 5.0, -1.5], dtype=np.float32)
    np.testing.assert_allclose(add_naive(x, y), np.array([5.0, 3.0, 2.0], dtype=np.float32))


@pytest.mark.torch
def test_add_torch_matches_naive() -> None:
    x = torch.randn((32,), dtype=torch.float32)
    y = torch.randn((32,), dtype=torch.float32)
    torch.testing.assert_close(add_torch(x, y), add_naive(x, y))


@pytest.mark.torch
def test_add_torch_compile_matches_naive() -> None:
    x = torch.randn((32,), dtype=torch.float32)
    y = torch.randn((32,), dtype=torch.float32)
    torch.testing.assert_close(add_torch_compile(x, y), add_naive(x, y))


@pytest.mark.jax
def test_add_jax_jit_matches_naive(require_jax: Any) -> None:
    _ = require_jax
    x = runtime.jax_random((32,), dtype="fp32", seed=0)
    y = runtime.jax_random((32,), dtype="fp32", seed=1)
    np.testing.assert_allclose(
        runtime.jax_to_numpy(add_jax_jit(x, y)),
        runtime.jax_to_numpy(add_naive(x, y)),
    )


@pytest.mark.triton
@pytest.mark.gpu
def test_add_triton_matches_naive(require_cuda: Any, require_triton: Any) -> None:
    torch = require_cuda
    _ = require_triton
    x = torch.randn((33,), device="cuda", dtype=torch.float32)
    y = torch.randn((33,), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(add_triton(x, y, block_shape=(16,)), add_naive(x, y))
