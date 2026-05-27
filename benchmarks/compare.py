"""Compare variants of one op against the same GPU roofline."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from benchmarks import _history
from benchmarks.roofline import (
    HardwarePeak,
    Roofline,
    analyze,
    arithmetic_intensity,
    peak_flops_for,
    profile,
    regime,
)
from benchmarks.runner import BenchResult, Variant, bench
from benchmarks.workload import Workload

BenchRow = tuple[str, BenchResult, Roofline]


def compare(
    workload: Workload,
    variants: Mapping[str, Callable[..., Any] | Variant],
    *,
    kwargs: dict[str, Any] | None = None,
    hw: HardwarePeak | None = None,
    warmup: int = 10,
    iters: int = 50,
    write_history: bool = True,
    config: dict[str, Any] | None = None,
) -> dict[str, Roofline]:
    """Bench each variant, print a roofline table, and write history JSON."""
    kwargs = kwargs or {}
    hw = hw if hw is not None else profile("rtx4090")
    normalized = [_normalize_variant(name, value) for name, value in variants.items()]

    results: dict[str, Roofline] = {}
    bench_results: list[BenchRow] = []
    for variant in normalized:
        br = bench(variant, args=workload.args, kwargs=kwargs, warmup=warmup, iters=iters)
        roof = analyze(
            flops=workload.flops,
            nbytes=workload.nbytes,
            seconds=br.median_s,
            hw=hw,
            flop_dtype=workload.flop_dtype,
        )
        results[variant.name] = roof
        bench_results.append((variant.name, br, roof))

    _print_table(
        workload.op,
        hw,
        workload.flops,
        workload.nbytes,
        workload.flop_dtype,
        bench_results,
    )
    _reject_unphysical_sol(bench_results)

    if write_history:
        _write_history(workload, hw, bench_results, config=config)

    return results


def _normalize_variant(name: str, value: Callable[..., Any] | Variant) -> Variant:
    if isinstance(value, Variant):
        return value
    return Variant(name=name, fn=value, framework="python")


def _print_table(
    op: str,
    hw: HardwarePeak,
    flops: int,
    nbytes: int,
    flop_dtype: str,
    rows: list[BenchRow],
) -> None:
    peak_flops = peak_flops_for(hw, flop_dtype)  # type: ignore[arg-type]
    ridge = peak_flops / hw.total_memory_bw
    intensity = arithmetic_intensity(flops, nbytes)
    print(f"\n=== {op} ({hw.name}, {hw.num_gpus} gpu(s)) ===")
    print(f"  flops={flops:.3e}  bytes={nbytes:.3e}  intensity={intensity:.3f} F/B")
    print(f"  ridge point={ridge:.1f} F/B  ->  {regime(intensity, ridge)}")
    print()
    print("  variant              med(us)   p99(us)    MFU%     BW%    binds        SoL%")
    print("  -----------------------------------------------------------------------------")
    for name, br, roof in rows:
        print(
            f"  {name:<18}"
            f"{br.median_s * 1e6:9.2f}"
            f"{br.p99_s * 1e6:10.2f}"
            f"{roof.mfu_pct * 100:8.1f}%"
            f"{roof.bw_pct * 100:8.1f}%"
            f"  {roof.binds:<12}"
            f"{roof.sol_pct * 100:7.1f}%"
        )


def _reject_unphysical_sol(rows: list[BenchRow]) -> None:
    bad = [(name, roof.sol_pct) for name, _, roof in rows if roof.sol_pct > 1.0]
    if not bad:
        return
    details = ", ".join(f"{name} {pct * 100:.1f}%" for name, pct in bad)
    raise RuntimeError(
        f"SoL > 100% is unphysical for variant(s): {details}. Check timing, "
        "hardware profile, FLOP counts, byte counts, and input sizes."
    )


def _write_history(
    workload: Workload,
    hw: HardwarePeak,
    rows: list[BenchRow],
    *,
    config: dict[str, Any] | None,
) -> None:
    out_dir = _history.HISTORY_DIR / workload.op
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    path = out_dir / f"{ts}.json"
    record = {
        "kind": "compare",
        "timestamp": ts,
        "git_sha": _history.git_sha(),
        "workload": {
            "op": workload.op,
            "flops": workload.flops,
            "nbytes": workload.nbytes,
            "flop_dtype": workload.flop_dtype,
        },
        "hardware": asdict(hw),
        "config": config or {},
        "variants": [
            {
                "name": name,
                "bench": asdict(br),
                "roofline": {
                    "mfu_pct": roof.mfu_pct,
                    "bw_pct": roof.bw_pct,
                    "sol_pct": roof.sol_pct,
                    "binds": roof.binds,
                    "seconds": roof.seconds,
                },
            }
            for name, br, roof in rows
        ],
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    if path.is_relative_to(_history.REPO_ROOT):
        display_path = path.relative_to(_history.REPO_ROOT)
    else:
        display_path = path
    print(f"\n  wrote {display_path}")
