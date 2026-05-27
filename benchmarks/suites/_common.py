"""Shared CLI scaffolding for op benchmark suites."""

from __future__ import annotations

import argparse

from benchmarks.roofline import PROFILES


def csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32", "tf32"], default="bf16")
    parser.add_argument("--gpu-profile", choices=sorted(PROFILES), default="rtx4090")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--block",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Block shape / Triton tile parameters, one int per axis.",
    )
    parser.add_argument(
        "--sweep-block",
        type=csv_ints,
        nargs="*",
        default=None,
        metavar="AXIS_LIST",
        help="Cartesian sweep over block/tile axes, one CSV per axis.",
    )
    return parser


def validate_block_shapes(
    args: argparse.Namespace,
    *,
    expected_axes: int,
    parser: argparse.ArgumentParser,
) -> None:
    if args.block is None:
        parser.error("suite must call parser.set_defaults(block=[...])")
    if len(args.block) != expected_axes:
        example = " ".join(["128"] * expected_axes)
        parser.error(
            f"--block expects {expected_axes} value(s), got {len(args.block)}. "
            f"Example: --block {example}"
        )
    if args.sweep_block is not None and len(args.sweep_block) != expected_axes:
        example = " ".join(["8,16"] * expected_axes)
        parser.error(
            f"--sweep-block expects {expected_axes} axis list(s), got {len(args.sweep_block)}. "
            f"Example: --sweep-block {example}"
        )
