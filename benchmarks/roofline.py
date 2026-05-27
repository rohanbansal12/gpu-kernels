"""GPU roofline math.

A roofline gives a physical lower bound for runtime from the hardware's compute
peak and global-memory bandwidth. The result is deliberately simple: pick the
peak matching the dtype, compute the memory floor, and report utilization
against whichever floor binds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FlopDtype = Literal["bf16", "fp16", "fp32", "tf32", "int8"]


@dataclass(frozen=True)
class HardwarePeak:
    """Sustained-peak model for one GPU or a homogeneous set of GPUs."""

    name: str
    bf16_flops: float
    fp16_flops: float
    fp32_flops: float
    tf32_flops: float
    int8_ops: float
    memory_bw: float
    num_gpus: int = 1

    @property
    def total_memory_bw(self) -> float:
        return self.memory_bw * self.num_gpus


def rtx_4090(num_gpus: int = 1) -> HardwarePeak:
    """Conservative starting profile for an RTX 4090.

    These are nominal reference-card peaks, not guaranteed sustained numbers.
    Prefer adding an explicit profile for the exact GPU used for serious runs.
    """
    return HardwarePeak(
        name="RTX 4090",
        bf16_flops=82.6e12,
        fp16_flops=82.6e12,
        fp32_flops=82.6e12,
        tf32_flops=82.6e12,
        int8_ops=330.3e12,
        memory_bw=1_008e9,
        num_gpus=num_gpus,
    )


def a100_80gb(num_gpus: int = 1) -> HardwarePeak:
    """Nominal A100 SXM 80GB profile, useful for cloud baselines."""
    return HardwarePeak(
        name="A100 SXM 80GB",
        bf16_flops=312e12,
        fp16_flops=312e12,
        fp32_flops=19.5e12,
        tf32_flops=156e12,
        int8_ops=624e12,
        memory_bw=2_039e9,
        num_gpus=num_gpus,
    )


def h100_80gb(num_gpus: int = 1) -> HardwarePeak:
    """Nominal H100 SXM 80GB profile, useful for cloud baselines."""
    return HardwarePeak(
        name="H100 SXM 80GB",
        bf16_flops=989e12,
        fp16_flops=989e12,
        fp32_flops=67e12,
        tf32_flops=494e12,
        int8_ops=1_979e12,
        memory_bw=3_350e9,
        num_gpus=num_gpus,
    )


PROFILES = {
    "rtx4090": rtx_4090,
    "a100_80gb": a100_80gb,
    "h100_80gb": h100_80gb,
}


def profile(name: str, *, num_gpus: int = 1) -> HardwarePeak:
    try:
        return PROFILES[name](num_gpus=num_gpus)
    except KeyError as e:
        raise ValueError(f"unknown GPU profile {name!r}; choices: {sorted(PROFILES)}") from e


def peak_flops_for(hw: HardwarePeak, flop_dtype: FlopDtype) -> float:
    if flop_dtype == "bf16":
        return hw.bf16_flops * hw.num_gpus
    if flop_dtype == "fp16":
        return hw.fp16_flops * hw.num_gpus
    if flop_dtype == "fp32":
        return hw.fp32_flops * hw.num_gpus
    if flop_dtype == "tf32":
        return hw.tf32_flops * hw.num_gpus
    return hw.int8_ops * hw.num_gpus


def arithmetic_intensity(flops: int, nbytes: int) -> float:
    return flops / nbytes if nbytes else float("inf")


Regime = Literal["compute-bound", "memory-bound"]


def regime(intensity: float, ridge_point: float) -> Regime:
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


@dataclass(frozen=True)
class Roofline:
    flops: int
    nbytes: int
    seconds: float
    hw: HardwarePeak
    flop_dtype: FlopDtype = "bf16"

    @property
    def peak_flops(self) -> float:
        return peak_flops_for(self.hw, self.flop_dtype)

    @property
    def compute_floor_s(self) -> float:
        return self.flops / self.peak_flops if self.peak_flops else 0.0

    @property
    def memory_floor_s(self) -> float:
        return self.nbytes / self.hw.total_memory_bw if self.hw.total_memory_bw else 0.0

    @property
    def sol_s(self) -> float:
        return max(self.compute_floor_s, self.memory_floor_s)

    @property
    def mfu_pct(self) -> float:
        return self.compute_floor_s / self.seconds if self.seconds else 0.0

    @property
    def bw_pct(self) -> float:
        return self.memory_floor_s / self.seconds if self.seconds else 0.0

    @property
    def sol_pct(self) -> float:
        return self.sol_s / self.seconds if self.seconds else 0.0

    @property
    def binds(self) -> Regime:
        return "compute-bound" if self.compute_floor_s >= self.memory_floor_s else "memory-bound"


def analyze(
    *,
    flops: int,
    nbytes: int,
    seconds: float,
    hw: HardwarePeak,
    flop_dtype: FlopDtype = "bf16",
) -> Roofline:
    return Roofline(flops=flops, nbytes=nbytes, seconds=seconds, hw=hw, flop_dtype=flop_dtype)
