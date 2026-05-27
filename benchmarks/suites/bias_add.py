"""Bench suite for the bias_add op."""

from __future__ import annotations

import argparse

from benchmarks.compare import compare
from benchmarks.roofline import profile
from benchmarks.runner import Variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.sweep import sweep
from benchmarks.workload import Workload
from gpu_kernels import runtime
from gpu_kernels.ops.bias_add import (
    bias_add_jax_jit,
    bias_add_torch,
    bias_add_torch_compile,
    bias_add_triton,
)
from gpu_kernels.ops.bias_add.triton_impl import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=2, parser=parser)

    torch = runtime.require_torch()
    device = runtime.torch_device(prefer_cuda=True)
    x_torch = runtime.torch_random(
        (args.rows, args.hidden), dtype=args.dtype, device=device, seed=0
    )
    bias_torch = runtime.torch_random((args.hidden,), dtype=args.dtype, device=device, seed=1)
    bytes_per_elem = torch.empty((), dtype=runtime.torch_dtype(args.dtype)).element_size()
    workload = Workload(
        op="bias_add",
        flops=args.rows * args.hidden,
        nbytes=(2 * args.rows * args.hidden + args.hidden) * bytes_per_elem,
        args=(x_torch, bias_torch),
        flop_dtype=args.dtype,
    )

    variants = {
        "torch": Variant(name="torch", fn=bias_add_torch, framework="torch"),
        "torch_compile": Variant(
            name="torch_compile",
            fn=bias_add_torch_compile,
            framework="torch",
        ),
    }

    if runtime.has_jax():
        x_jax = runtime.jax_random((args.rows, args.hidden), dtype=args.dtype, seed=0)
        bias_jax = runtime.jax_random((args.hidden,), dtype=args.dtype, seed=1)
        variants["jax_jit"] = Variant(
            name="jax_jit",
            fn=bias_add_jax_jit,
            framework="jax",
            args=(x_jax, bias_jax),
        )

    block_shape = tuple(args.block)
    hw = profile(args.gpu_profile)
    if args.sweep_block is not None:
        if not runtime.has_triton() or not runtime.torch_cuda_available():
            parser.error("--sweep-block for bias_add requires Triton and CUDA")
        block_ms, block_ns = args.sweep_block

        def variant_factory(config: dict[str, int]) -> Variant:
            block = (config["block_m"], config["block_n"])
            return Variant(
                name=f"triton_b{block[0]}x{block[1]}",
                fn=bias_add_triton,
                framework="torch",
                kwargs={"block_shape": block},
            )

        sweep(
            workload,
            axes={"block_m": list(block_ms), "block_n": list(block_ns)},
            variant_factory=variant_factory,
            hw=hw,
            warmup=args.warmup,
            iters=args.iters,
        )
        return

    if runtime.has_triton() and runtime.torch_cuda_available():
        variants["triton"] = Variant(
            name="triton",
            fn=bias_add_triton,
            framework="torch",
            kwargs={"block_shape": block_shape},
        )

    compare(
        workload,
        variants,
        hw=hw,
        warmup=args.warmup,
        iters=args.iters,
        config={
            "rows": args.rows,
            "hidden": args.hidden,
            "dtype": args.dtype,
            "device": str(device),
            "block_shape": list(block_shape),
        },
    )


if __name__ == "__main__":
    main()
