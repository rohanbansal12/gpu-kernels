from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from gpu_kernels import runtime
from gpu_kernels.ops.silu import (
    silu_jax_jit,
    silu_naive,
    silu_torch,
    silu_torch_compile,
    silu_triton,
)


def test_silu_naive_numpy() -> None:
    x = np.array([1.0, -2.0, 0.0, 3.5], dtype=np.float32)
    expected = x / (1 + np.exp(-x))
    np.testing.assert_allclose(silu_naive(x), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.torch
def test_silu_torch_matches_naive(require_torch: Any) -> None:
    torch = require_torch
    x = torch.randn((32,), dtype=torch.float32)
    torch.testing.assert_close(silu_torch(x), silu_naive(x))


@pytest.mark.torch
def test_silu_torch_compile_matches_naive(require_torch: Any) -> None:
    torch = require_torch
    x = torch.randn((32,), dtype=torch.float32)
    torch.testing.assert_close(silu_torch_compile(x), silu_naive(x))


@pytest.mark.jax
def test_silu_jax_jit_matches_naive(require_jax: Any) -> None:
    _ = require_jax
    x = runtime.jax_random((32,), dtype="fp32", seed=0)
    np.testing.assert_allclose(
        runtime.jax_to_numpy(silu_jax_jit(x)),
        runtime.jax_to_numpy(silu_naive(x)),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.triton
@pytest.mark.gpu
def test_silu_triton_matches_naive(require_cuda: Any, require_triton: Any) -> None:
    torch = require_cuda
    _ = require_triton
    x = torch.randn((33,), device="cuda", dtype=torch.float32)
    torch.testing.assert_close(
        silu_triton(x, block_shape=(16,)),
        silu_naive(x),
        rtol=1e-6,
        atol=1e-6,
    )
