"""Shared helpers for benchmark history writers."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DIR = REPO_ROOT / "bench_history"


def git_sha() -> str | None:
    """Resolve HEAD's SHA, with a "-dirty" suffix when inputs changed.

    History records are most useful when tied to a tree state. If this folder
    is not a git repo yet, return ``None`` and still allow local benchmarking.
    """
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if head.returncode != 0:
            return None
        sha = head.stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                ".",
                ":(exclude)bench_history",
                ":(exclude)tests",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if status.returncode == 0 and status.stdout.strip():
            sha = f"{sha}-dirty"
        return sha
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
