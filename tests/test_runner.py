from __future__ import annotations

from benchmarks.runner import Variant, bench


def test_bench_python_variant() -> None:
    result = bench(Variant(name="inc", fn=lambda x: x + 1), args=(1,), warmup=1, iters=2)
    assert result.name == "inc"
    assert result.warmup_iters == 1
    assert result.timed_iters == 2
    assert len(result.times_s) == 2
    assert result.median_s >= 0
