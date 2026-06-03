"""Bench suite for the silu_mul op."""

from __future__ import annotations

import argparse

import torch

from benchmarks.compare import compare
from benchmarks.roofline import profile
from benchmarks.runner import Variant
from benchmarks.suites._common import base_parser, validate_block_shapes
from benchmarks.sweep import sweep
from benchmarks.workload import Workload
from gpu_kernels import runtime
from gpu_kernels.ops.silu_mul import (
    silu_mul_jax_jit,
    silu_mul_torch,
    silu_mul_torch_compile,
    silu_mul_triton,
)
from gpu_kernels.ops.silu_mul.triton_impl import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument(
        "--n",
        type=int,
        default=256 * 1024 * 1024,
        help="number of elements; default is large enough for memory-bandwidth timing",
    )
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=1, parser=parser)

    device = runtime.torch_device(prefer_cuda=True)
    x_torch = runtime.torch_random((args.n,), dtype=args.dtype, device=device, seed=0)
    gate_torch = runtime.torch_random((args.n,), dtype=args.dtype, device=device, seed=1)
    bytes_per_elem = torch.empty((), dtype=runtime.torch_dtype(args.dtype)).element_size()
    # Count sigmoid as exp + add + div, plus SiLU multiply and gate multiply.
    flops = 5 * args.n
    workload = Workload(
        op="silu_mul",
        flops=flops,
        nbytes=3 * args.n * bytes_per_elem,
        args=(x_torch, gate_torch),
        flop_dtype=args.dtype,
    )

    variants = {
        "torch": Variant(name="torch", fn=silu_mul_torch, framework="torch"),
        "torch_compile": Variant(
            name="torch_compile",
            fn=silu_mul_torch_compile,
            framework="torch",
        ),
    }

    if runtime.has_jax():
        x_jax = runtime.jax_random((args.n,), dtype=args.dtype, seed=0)
        gate_jax = runtime.jax_random((args.n,), dtype=args.dtype, seed=1)
        variants["jax_jit"] = Variant(
            name="jax_jit",
            fn=silu_mul_jax_jit,
            framework="jax",
            args=(x_jax, gate_jax),
        )

    block_shape = tuple(args.block)
    hw = profile(args.gpu_profile)
    if args.sweep_block is not None:
        if not runtime.has_triton() or not runtime.torch_cuda_available():
            parser.error("--sweep-block for silu_mul requires Triton and CUDA")
        (block_sizes,) = args.sweep_block

        def variant_factory(config: dict[str, int]) -> Variant:
            block = (config["block_size"],)
            return Variant(
                name=f"triton_b{block[0]}",
                fn=silu_mul_triton,
                framework="torch",
                kwargs={"block_shape": block},
            )

        sweep(
            workload,
            axes={"block_size": list(block_sizes)},
            variant_factory=variant_factory,
            hw=hw,
            warmup=args.warmup,
            iters=args.iters,
        )
        return

    if runtime.has_triton() and runtime.torch_cuda_available():
        variants["triton"] = Variant(
            name="triton",
            fn=silu_mul_triton,
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
            "n": args.n,
            "dtype": args.dtype,
            "device": str(device),
            "block_shape": list(block_shape),
        },
    )


if __name__ == "__main__":
    main()
