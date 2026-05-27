from __future__ import annotations

import argparse

import pytest
from benchmarks.suites._common import base_parser, csv_ints, validate_block_shapes


def test_csv_ints() -> None:
    assert csv_ints("8,16,32") == (8, 16, 32)


def test_validate_block_shapes_accepts_expected_axes() -> None:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.set_defaults(block=[128, 256])
    args = parser.parse_args([])
    validate_block_shapes(args, expected_axes=2, parser=parser)


def test_validate_block_shapes_rejects_wrong_axes() -> None:
    parser = argparse.ArgumentParser(parents=[base_parser()])
    parser.set_defaults(block=[128])
    args = parser.parse_args([])
    with pytest.raises(SystemExit):
        validate_block_shapes(args, expected_axes=2, parser=parser)
