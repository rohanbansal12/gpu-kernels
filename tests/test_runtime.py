from __future__ import annotations

import pytest

from gpu_kernels import runtime


def test_available_frameworks_reports_known_keys() -> None:
    assert set(runtime.available_frameworks()) == {"torch", "jax", "triton"}


def test_has_package_false_for_missing_name() -> None:
    assert not runtime.has_package("definitely_not_a_real_kernel_package")


def test_torch_dtype_requires_torch_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> object:
        if name == "torch":
            raise ImportError("missing torch")
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(runtime, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="Torch is not installed"):
        runtime.torch_dtype("bf16")
