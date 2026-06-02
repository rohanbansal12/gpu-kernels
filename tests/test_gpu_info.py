from __future__ import annotations

from benchmarks.gpu_info import (
    profile_name_from_device_name,
    profile_name_from_modal_request,
    roofline_profile_for,
    validate_modal_gpu,
)


def test_profile_name_from_modal_request_handles_common_modal_gpu_names() -> None:
    assert profile_name_from_modal_request("H100!") == "h100"


def test_profile_name_from_device_name_supports_selected_gpu_set() -> None:
    assert profile_name_from_device_name("NVIDIA H100 80GB HBM3") == "h100"


def test_roofline_profile_prefers_actual_supported_device_over_request_hint() -> None:
    assert (
        roofline_profile_for(
            requested_gpu="H100!",
            device_names=["NVIDIA H100 80GB HBM3"],
            nvidia_smi=None,
        )
        == "h100"
    )


def test_validate_gpu_rejects_non_selected_modal_gpus() -> None:
    assert validate_modal_gpu("H100!") == "H100!"

    for gpu in ("H100", "H200", "A100-80GB", "RTX-PRO-6000", "L4"):
        try:
            validate_modal_gpu(gpu)
        except ValueError as e:
            assert "unsupported Modal GPU" in str(e)
        else:
            raise AssertionError(f"expected {gpu} to be rejected")
