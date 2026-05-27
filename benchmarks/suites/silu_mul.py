"""Bench suite scaffold for the silu_mul op."""

from __future__ import annotations

import argparse

from benchmarks.suites._common import base_parser, validate_block_shapes
from gpu_kernels.ops.silu_mul.triton_impl import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--n", type=int, default=256 * 1024 * 1024)
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=1, parser=parser)
    raise NotImplementedError("implement benchmarks.suites.silu_mul")


if __name__ == "__main__":
    main()
