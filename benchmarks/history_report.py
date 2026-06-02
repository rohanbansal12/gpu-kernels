"""Summarize benchmark history records for one op across GPUs and configs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from benchmarks import _history


@dataclass(frozen=True, slots=True)
class HistoryRow:
    op: str
    timestamp: str
    git_sha: str | None
    gpu: str
    profile: str
    device: str
    dtype: str
    config: dict[str, Any]
    variant: str
    median_us: float
    p99_us: float
    gb_s: float
    tflops: float
    mfu_pct: float
    bw_pct: float
    sol_pct: float
    timing: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", help="op name under bench_history/<op>")
    parser.add_argument("--history-dir", type=Path, default=_history.HISTORY_DIR)
    parser.add_argument("--variant", help="only show one variant")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="show only the latest record for each GPU/config/variant",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="show only the best SoL row for each GPU/config",
    )
    parser.add_argument("--json", action="store_true", help="emit rows as JSON")
    args = parser.parse_args()

    rows = load_rows(args.op, history_dir=args.history_dir)
    if args.variant:
        rows = [row for row in rows if row.variant == args.variant]
    if args.latest:
        rows = latest_rows(rows)
    if args.best:
        rows = best_rows(rows)

    if args.json:
        print(json.dumps([row_to_dict(row) for row in rows], indent=2))
        return

    print_report(args.op, rows)


def load_rows(op: str, *, history_dir: Path = _history.HISTORY_DIR) -> list[HistoryRow]:
    rows: list[HistoryRow] = []
    for path in sorted((history_dir / op).glob("*.json")):
        record = json.loads(path.read_text())
        rows.extend(_rows_from_record(path, record))
    return rows


def latest_rows(rows: list[HistoryRow]) -> list[HistoryRow]:
    latest: dict[tuple[str, str, str, str], HistoryRow] = {}
    for row in rows:
        key = (row.gpu, _config_label(row.config), row.dtype, row.variant)
        if key not in latest or row.timestamp > latest[key].timestamp:
            latest[key] = row
    return sorted(latest.values(), key=_sort_key)


def best_rows(rows: list[HistoryRow]) -> list[HistoryRow]:
    best: dict[tuple[str, str, str], HistoryRow] = {}
    for row in rows:
        key = (row.gpu, _config_label(row.config), row.dtype)
        if key not in best or row.sol_pct > best[key].sol_pct:
            best[key] = row
    return sorted(best.values(), key=_sort_key)


def print_report(op: str, rows: list[HistoryRow]) -> None:
    if not rows:
        print(f"No benchmark history for {op!r}.")
        return

    print(f"\n=== {op} benchmark history ({len(rows)} row(s)) ===")
    print(
        "timestamp             gpu/profile            dtype  config"
        "                         variant          med(us)   p99(us)      GB/s   TFLOP/s"
        "    BW%   SoL%"
    )
    print("-" * 128)
    for row in rows:
        gpu_label = _truncate(f"{row.gpu}/{row.profile}", 22)
        config = _truncate(_config_label(row.config), 28)
        print(
            f"{row.timestamp[:17]:<21}"
            f"{gpu_label:<23}"
            f"{row.dtype:<7}"
            f"{config:<31}"
            f"{row.variant:<16}"
            f"{row.median_us:8.2f}"
            f"{row.p99_us:10.2f}"
            f"{row.gb_s:10.1f}"
            f"{row.tflops:10.2f}"
            f"{row.bw_pct:7.1f}%"
            f"{row.sol_pct:7.1f}%"
        )

    print("\nBest variant by GPU/config:")
    print(
        "gpu/profile            dtype  config                         "
        "variant             GB/s   SoL%"
    )
    print("-" * 92)
    for row in best_rows(rows):
        gpu_label = _truncate(f"{row.gpu}/{row.profile}", 22)
        config = _truncate(_config_label(row.config), 28)
        print(
            f"{gpu_label:<23}"
            f"{row.dtype:<7}"
            f"{config:<31}"
            f"{row.variant:<16}"
            f"{row.gb_s:10.1f}"
            f"{row.sol_pct:7.1f}%"
        )


def row_to_dict(row: HistoryRow) -> dict[str, Any]:
    return {
        "op": row.op,
        "timestamp": row.timestamp,
        "git_sha": row.git_sha,
        "gpu": row.gpu,
        "profile": row.profile,
        "device": row.device,
        "dtype": row.dtype,
        "config": row.config,
        "variant": row.variant,
        "median_us": row.median_us,
        "p99_us": row.p99_us,
        "gb_s": row.gb_s,
        "tflops": row.tflops,
        "mfu_pct": row.mfu_pct,
        "bw_pct": row.bw_pct,
        "sol_pct": row.sol_pct,
        "timing": row.timing,
        "path": str(row.path),
    }


def _rows_from_record(path: Path, record: dict[str, Any]) -> list[HistoryRow]:
    workload = record["workload"]
    hardware = record["hardware"]
    config = record.get("config", {})
    modal = record.get("modal", {})
    gpu_info = modal.get("gpu_info", {})
    devices = gpu_info.get("devices", [])
    device = devices[0]["name"] if devices else str(config.get("device", "unknown"))
    gpu = modal.get("requested_gpu") or hardware["name"]
    profile = gpu_info.get("roofline_profile") or hardware["name"]

    rows = []
    for variant in record["variants"]:
        bench = variant["bench"]
        roofline = variant["roofline"]
        seconds = roofline["seconds"]
        rows.append(
            HistoryRow(
                op=workload["op"],
                timestamp=record["timestamp"],
                git_sha=record.get("git_sha"),
                gpu=gpu,
                profile=profile,
                device=device,
                dtype=workload["flop_dtype"],
                config=config,
                variant=variant["name"],
                median_us=seconds * 1e6,
                p99_us=_nearest_rank_percentile(bench["times_s"], 0.99) * 1e6,
                gb_s=roofline.get("gb_s", _gb_per_s(workload["nbytes"], seconds)),
                tflops=roofline.get("tflops", _tflops_per_s(workload["flops"], seconds)),
                mfu_pct=roofline["mfu_pct"] * 100,
                bw_pct=roofline["bw_pct"] * 100,
                sol_pct=roofline["sol_pct"] * 100,
                timing=bench["timing"],
                path=path,
            )
        )
    return rows


def _sort_key(row: HistoryRow) -> tuple[str, str, str, str]:
    return (row.gpu, _config_label(row.config), row.variant, row.timestamp)


def _config_label(config: dict[str, Any]) -> str:
    parts = []
    preferred_keys = ("n", "rows", "hidden", "axis", "eps", "approximate", "block_shape")
    keys = [key for key in preferred_keys if key in config]
    keys.extend(sorted(key for key in config if key not in preferred_keys))
    for key in keys:
        value = config[key]
        if key in {"device", "dtype"}:
            continue
        if isinstance(value, list):
            value = "x".join(str(item) for item in value)
        label = "block" if key == "block_shape" else key
        parts.append(f"{label}={value}")
    return " ".join(parts) if parts else "-"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: width - 1] + "…"


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, ceil(percentile * len(values)) - 1))
    return sorted(values)[index]


def _gb_per_s(nbytes: int, seconds: float) -> float:
    return nbytes / seconds / 1e9 if seconds else 0.0


def _tflops_per_s(flops: int, seconds: float) -> float:
    return flops / seconds / 1e12 if seconds else 0.0


if __name__ == "__main__":
    main()
