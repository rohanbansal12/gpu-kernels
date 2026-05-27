from __future__ import annotations

from benchmarks.suites.scale import _make_parser


def test_scale_suite_defaults() -> None:
    parser = _make_parser()
    args = parser.parse_args([])
    assert args.n == 256 * 1024 * 1024
    assert args.block == [1024]
    assert args.dtype == "bf16"
