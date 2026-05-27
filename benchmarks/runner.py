"""Benchmark runner with framework-aware synchronization.

Torch/Triton work is timed with CUDA events when CUDA tensors are present and
falls back to wall-clock timing otherwise. JAX work uses ``block_until_ready``.
The fallback makes harness tests run on CPU-only machines; GPU performance runs
should use CUDA tensors.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

Framework = Literal["python", "torch", "jax"]


@dataclass(frozen=True, slots=True)
class Variant:
    """A callable plus the synchronization policy needed to benchmark it."""

    name: str
    fn: Callable[..., Any]
    framework: Framework = "python"
    args: Sequence[Any] | None = None
    kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class BenchResult:
    name: str
    times_s: list[float]
    warmup_iters: int
    timed_iters: int
    timing: Literal["cuda_event", "wall"] = "wall"

    @property
    def median_s(self) -> float:
        return statistics.median(self.times_s)

    @property
    def p99_s(self) -> float:
        if len(self.times_s) < 2:
            return max(self.times_s)
        return statistics.quantiles(self.times_s, n=100)[98]

    @property
    def min_s(self) -> float:
        return min(self.times_s)

    @property
    def stdev_s(self) -> float:
        return statistics.stdev(self.times_s) if len(self.times_s) >= 2 else 0.0


def bench(
    variant: Variant,
    args: Sequence[Any] = (),
    kwargs: dict[str, Any] | None = None,
    *,
    warmup: int = 10,
    iters: int = 50,
) -> BenchResult:
    """Run one variant and collect per-call timings."""
    args = variant.args if variant.args is not None else args
    kwargs = variant.kwargs if variant.kwargs is not None else kwargs
    kwargs = kwargs or {}
    for _ in range(warmup):
        out = variant.fn(*args, **kwargs)
        _sync(variant.framework, out)

    if variant.framework == "torch" and _torch_cuda_available(args):
        return _bench_torch_cuda_event(variant, args, kwargs, warmup=warmup, iters=iters)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = variant.fn(*args, **kwargs)
        _sync(variant.framework, out)
        times.append(time.perf_counter() - start)
    return BenchResult(
        name=variant.name,
        times_s=times,
        warmup_iters=warmup,
        timed_iters=iters,
        timing="wall",
    )


def _bench_torch_cuda_event(
    variant: Variant,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    warmup: int,
    iters: int,
) -> BenchResult:
    torch = import_module("torch")

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = variant.fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        _keepalive(out)
        times.append(start.elapsed_time(end) / 1_000.0)

    return BenchResult(
        name=variant.name,
        times_s=times,
        warmup_iters=warmup,
        timed_iters=iters,
        timing="cuda_event",
    )


def _sync(framework: Framework, out: Any) -> None:
    if framework == "jax":
        _jax_block_until_ready(out)
    elif framework == "torch":
        _torch_synchronize_if_available()
        _keepalive(out)


def _jax_block_until_ready(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
        return
    if isinstance(value, tuple | list):
        for item in value:
            _jax_block_until_ready(item)
    elif isinstance(value, dict):
        for item in value.values():
            _jax_block_until_ready(item)


def _torch_synchronize_if_available() -> None:
    try:
        torch = import_module("torch")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _torch_cuda_available(args: Sequence[Any]) -> bool:
    try:
        torch = import_module("torch")
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    return any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in _flatten(args))


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, tuple | list):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(_flatten(item))
        return out
    return [value]


def _keepalive(value: Any) -> None:
    # Hook for symmetry with JAX. Keeping a reference until after synchronization
    # avoids over-eager cleanup in tiny benchmark snippets.
    _ = value
