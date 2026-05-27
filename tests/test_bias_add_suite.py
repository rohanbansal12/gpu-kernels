from __future__ import annotations

from benchmarks.suites.bias_add import _make_parser


def test_bias_add_suite_defaults() -> None:
    parser = _make_parser()
    args = parser.parse_args([])
    assert args.rows == 32768
    assert args.hidden == 8192
    assert args.block == [16, 1024]
    assert args.dtype == "bf16"
