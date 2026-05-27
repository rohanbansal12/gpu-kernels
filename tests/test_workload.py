from __future__ import annotations

from benchmarks.workload import Workload


def test_workload_stores_accounting() -> None:
    workload = Workload(op="scale", flops=8, nbytes=32, args=(1, 2), flop_dtype="fp32")
    assert workload.op == "scale"
    assert workload.args == (1, 2)
    assert workload.flops == 8
    assert workload.nbytes == 32
    assert workload.flop_dtype == "fp32"
