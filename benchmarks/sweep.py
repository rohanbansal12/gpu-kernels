"""Small Cartesian sweep helper for Triton-style meta-parameters."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

from benchmarks.compare import compare
from benchmarks.roofline import HardwarePeak
from benchmarks.runner import Variant
from benchmarks.workload import Workload


def sweep(
    workload: Workload,
    *,
    axes: dict[str, Iterable[Any]],
    variant_factory: Callable[[dict[str, Any]], Variant],
    is_valid: Callable[[dict[str, Any]], bool] | None = None,
    hw: HardwarePeak | None = None,
    warmup: int = 10,
    iters: int = 50,
) -> dict[str, Any]:
    """Run ``compare`` for every valid Cartesian product and return the winner."""
    names = list(axes)
    configs = [
        dict(zip(names, values, strict=True)) for values in product(*(axes[n] for n in names))
    ]
    valid_configs = [cfg for cfg in configs if is_valid is None or is_valid(cfg)]
    if not valid_configs:
        raise ValueError("sweep has no valid configurations")

    best: dict[str, Any] | None = None
    for cfg in valid_configs:
        variant = variant_factory(cfg)
        results = compare(
            workload,
            {variant.name: variant},
            hw=hw,
            warmup=warmup,
            iters=iters,
            config=cfg,
        )
        roof = results[variant.name]
        if best is None or roof.sol_pct > best["sol_pct"]:
            best = {"config": cfg, "variant": variant.name, "sol_pct": roof.sol_pct}

    assert best is not None
    print(f"\nBest: {best['variant']} config={best['config']} SoL={best['sol_pct'] * 100:.1f}%")
    return best
