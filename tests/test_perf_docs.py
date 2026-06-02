from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_SECTIONS = (
    "## Target",
    "## Correctness",
    "## Results",
    "## Interpretation",
    "## Next",
)


def test_all_ops_use_perf_template_sections() -> None:
    perf_files = sorted((REPO_ROOT / "src/gpu_kernels/ops").glob("*/PERF.md"))

    assert perf_files
    for path in perf_files:
        text = path.read_text()
        missing = [section for section in REQUIRED_SECTIONS if section not in text]
        assert not missing, f"{path.relative_to(REPO_ROOT)} missing {missing}"
        assert "| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |" in text
        assert "Target:" not in text
        assert "Current:" not in text
