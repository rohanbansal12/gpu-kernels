from __future__ import annotations

import time

from benchmarks import _history
from benchmarks.compare import compare
from benchmarks.roofline import profile
from benchmarks.runner import Variant
from benchmarks.workload import Workload


def test_compare_runs_and_writes_history(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_history, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(_history, "git_sha", lambda: "test-sha")

    def slow_identity(x: int) -> int:
        time.sleep(0.001)
        return x

    results = compare(
        Workload(op="toy", flops=1, nbytes=1, args=(3,), flop_dtype="fp32"),
        {"python": Variant(name="python", fn=slow_identity)},
        hw=profile("rtx4090"),
        warmup=1,
        iters=2,
    )

    assert "python" in results
    records = list((tmp_path / "toy").glob("*.json"))
    assert len(records) == 1


def test_git_sha_env_override(monkeypatch) -> None:
    monkeypatch.setenv("GPU_KERNELS_GIT_SHA", "remote-sha-dirty")

    assert _history.git_sha() == "remote-sha-dirty"
