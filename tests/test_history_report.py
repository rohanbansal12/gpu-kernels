from __future__ import annotations

import json

from benchmarks.history_report import best_rows, latest_rows, load_rows


def test_history_report_loads_and_summarizes_rows(tmp_path) -> None:
    op_dir = tmp_path / "scale"
    op_dir.mkdir()
    (op_dir / "20260101T000000_000000Z.json").write_text(
        json.dumps(
            {
                "kind": "compare",
                "timestamp": "20260101T000000_000000Z",
                "git_sha": "abc",
                "workload": {
                    "op": "scale",
                    "flops": 1_000,
                    "nbytes": 10_000,
                    "flop_dtype": "fp32",
                },
                "hardware": {
                    "name": "A100 SXM 80GB",
                    "memory_bw": 2_000_000,
                },
                "config": {
                    "n": 1024,
                    "dtype": "fp32",
                    "device": "cuda",
                    "block_shape": [1024],
                },
                "variants": [
                    {
                        "name": "torch",
                        "bench": {
                            "name": "torch",
                            "times_s": [0.001, 0.002, 0.003],
                            "warmup_iters": 1,
                            "timed_iters": 3,
                            "timing": "cuda_event",
                        },
                        "roofline": {
                            "mfu_pct": 0.1,
                            "bw_pct": 0.2,
                            "sol_pct": 0.2,
                            "binds": "memory-bound",
                            "seconds": 0.002,
                        },
                    },
                    {
                        "name": "triton",
                        "bench": {
                            "name": "triton",
                            "times_s": [0.001, 0.001, 0.001],
                            "warmup_iters": 1,
                            "timed_iters": 3,
                            "timing": "cuda_event",
                        },
                        "roofline": {
                            "mfu_pct": 0.2,
                            "bw_pct": 0.4,
                            "sol_pct": 0.4,
                            "binds": "memory-bound",
                            "seconds": 0.001,
                        },
                    },
                ],
                "modal": {
                    "requested_gpu": "A100-80GB",
                    "gpu_info": {
                        "roofline_profile": "a100_80gb",
                        "devices": [{"name": "NVIDIA A100-SXM4-80GB"}],
                    },
                },
            }
        )
        + "\n"
    )

    rows = load_rows("scale", history_dir=tmp_path)

    assert len(rows) == 2
    assert rows[0].gpu == "A100-80GB"
    assert rows[0].profile == "a100_80gb"
    assert rows[0].p99_us == 3000.0
    assert rows[0].gb_s == 0.005
    assert [row.variant for row in best_rows(rows)] == ["triton"]
    assert len(latest_rows(rows)) == 2
