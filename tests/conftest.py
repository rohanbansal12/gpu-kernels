from __future__ import annotations

import pytest

from gpu_kernels import runtime


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "torch: requires PyTorch")
    config.addinivalue_line("markers", "jax: requires JAX")
    config.addinivalue_line("markers", "triton: requires Triton")


@pytest.fixture
def require_torch() -> object:
    if not runtime.has_torch():
        pytest.skip("PyTorch is not installed; run `uv sync --extra local --group dev`.")
    return runtime.require_torch()


@pytest.fixture
def require_jax() -> object:
    if not runtime.has_jax():
        pytest.skip("JAX is not installed; run `uv sync --extra local --group dev`.")
    return runtime.require_jax()


@pytest.fixture
def require_triton() -> object:
    if not runtime.has_triton():
        pytest.skip("Triton is not installed; run `uv sync --extra cuda --group dev`.")
    return runtime.require_triton()


@pytest.fixture
def require_cuda() -> object:
    if not runtime.torch_cuda_available():
        pytest.skip("CUDA is not available through PyTorch.")
    return runtime.require_torch()
