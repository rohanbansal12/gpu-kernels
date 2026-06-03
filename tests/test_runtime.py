from __future__ import annotations

from gpu_kernels import runtime


def test_available_frameworks_reports_known_keys() -> None:
    assert set(runtime.available_frameworks()) == {"torch", "jax", "triton"}


def test_has_package_false_for_missing_name() -> None:
    assert not runtime.has_package("definitely_not_a_real_kernel_package")

def test_torch_is_always_available() -> None:
    assert runtime.has_torch()
    assert runtime.torch_dtype("bf16") is not None
