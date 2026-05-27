from __future__ import annotations

import pytest
from benchmarks.roofline import analyze, arithmetic_intensity, profile, regime


def test_profile_lookup_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown GPU profile"):
        profile("not-a-gpu")


def test_roofline_classifies_memory_bound_work() -> None:
    hw = profile("rtx4090")
    roof = analyze(flops=1_000, nbytes=1_000_000, seconds=1e-3, hw=hw, flop_dtype="bf16")
    assert roof.binds == "memory-bound"
    assert roof.bw_pct > roof.mfu_pct
    assert roof.sol_pct == roof.bw_pct


def test_regime_tie_breaks_to_compute_bound() -> None:
    assert regime(arithmetic_intensity(10, 2), ridge_point=5) == "compute-bound"
