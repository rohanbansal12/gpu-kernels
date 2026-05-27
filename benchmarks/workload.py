"""``Workload``: the "what is being benched" bundle shared by harness code."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from benchmarks.roofline import FlopDtype


@dataclass(frozen=True, slots=True)
class Workload:
    """The inputs and accounting for one benchmarked operation.

    Attributes:
        op: Op name, used in table labels and history paths.
        flops: Floating-point or integer ops per call.
        nbytes: Bytes moved to/from global memory per call.
        args: Input tensors/arrays passed to each variant.
        flop_dtype: Which peak to compare against for roofline math.
    """

    op: str
    flops: int
    nbytes: int
    args: Sequence[Any] = ()
    flop_dtype: FlopDtype = "bf16"
