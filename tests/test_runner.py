from __future__ import annotations

from benchmarks.runner import BenchResult, Variant, bench


def test_bench_python_variant() -> None:
    result = bench(Variant(name="inc", fn=lambda x: x + 1), args=(1,), warmup=1, iters=2)
    assert result.name == "inc"
    assert result.warmup_iters == 1
    assert result.timed_iters == 2
    assert len(result.times_s) == 2
    assert result.median_s >= 0


def test_p99_uses_observed_nearest_rank_value() -> None:
    result = BenchResult(
        name="toy",
        times_s=[1.0, 2.0, 100.0],
        warmup_iters=0,
        timed_iters=3,
    )

    assert result.p99_s == 100.0
