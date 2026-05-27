"""Bench suite scaffold for the rmsnorm op."""

from __future__ import annotations

import argparse

from benchmarks.suites._common import base_parser, validate_block_shapes
from gpu_kernels.ops.rmsnorm.triton_impl import DEFAULT_BLOCK


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=8192)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.set_defaults(block=list(DEFAULT_BLOCK))
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    validate_block_shapes(args, expected_axes=2, parser=parser)
    raise NotImplementedError("implement benchmarks.suites.rmsnorm")


if __name__ == "__main__":
    main()
