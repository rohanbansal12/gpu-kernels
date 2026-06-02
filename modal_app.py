"""Modal entrypoints for remote GPU correctness and benchmark runs.

Examples:

    uv run --extra cloud modal run modal_app.py::gpu_info --gpu H100!
    uv run --extra cloud modal run modal_app.py::correctness --op add
    uv run --extra cloud modal run modal_app.py::benchmark --gpu H100! --op add
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import modal
from benchmarks import _history
from benchmarks.gpu_info import SUPPORTED_MODAL_GPUS, validate_modal_gpu

APP_NAME = "gpu-kernels-lab"
REPO_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/repo")
REMOTE_HISTORY_DIR = Path("/tmp/gpu-kernels-bench-history")
DEFAULT_GPU = "H100!"
BENCHMARK_OPS = (
    "add",
    "bias_add",
    "gelu",
    "layernorm",
    "rmsnorm",
    "row_max",
    "row_sum",
    "scale",
    "silu",
    "silu_mul",
    "softmax",
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=2.0",
        "pytest>=8.3",
        "torch>=2.6",
        "jax[cuda12]>=0.5",
        "triton>=3.0",
    )
    .env({"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}"})
    .add_local_dir(str(REPO_ROOT / "src"), remote_path=str(REMOTE_ROOT / "src"))
    .add_local_dir(str(REPO_ROOT / "benchmarks"), remote_path=str(REMOTE_ROOT / "benchmarks"))
    .add_local_dir(str(REPO_ROOT / "tests"), remote_path=str(REMOTE_ROOT / "tests"))
    .add_local_file(
        str(REPO_ROOT / "pyproject.toml"),
        remote_path=str(REMOTE_ROOT / "pyproject.toml"),
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(timeout=20 * 60, cpu=4.0, memory=8192)
def _gpu_info(requested_gpu: str | None) -> dict[str, Any]:
    from benchmarks.gpu_info import collect_gpu_info

    return collect_gpu_info(requested_gpu=requested_gpu).to_dict()


@app.function(timeout=30 * 60, cpu=4.0, memory=16384)
def _run_correctness(requested_gpu: str | None, pytest_args: list[str]) -> dict[str, Any]:
    from benchmarks.gpu_info import collect_gpu_info

    gpu_info = collect_gpu_info(requested_gpu=requested_gpu).to_dict()
    command = [sys.executable, "-m", "pytest", *pytest_args]
    result = _run(command, env={"PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}"})
    result["gpu_info"] = gpu_info
    return result


@app.function(timeout=60 * 60, cpu=4.0, memory=16384)
def _run_benchmark(
    requested_gpu: str | None,
    op: str,
    suite_args: list[str],
    git_sha: str | None,
) -> dict[str, Any]:
    from benchmarks.gpu_info import collect_gpu_info

    gpu_info = collect_gpu_info(requested_gpu=requested_gpu).to_dict()
    profile_name = gpu_info.get("roofline_profile")
    if "--gpu-profile" not in suite_args and profile_name is None:
        return {
            "returncode": 2,
            "stdout": "",
            "stderr": (
                "Could not infer a roofline profile for this GPU. "
                'Pass --suite-args "--gpu-profile <profile> ..." explicitly.'
            ),
            "gpu_info": gpu_info,
            "history": [],
        }

    resolved_args = list(suite_args)
    if "--gpu-profile" not in resolved_args:
        resolved_args.extend(["--gpu-profile", str(profile_name)])

    command = [sys.executable, "-m", f"benchmarks.suites.{op}", *resolved_args]
    env = {
        "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT}",
        "GPU_KERNELS_HISTORY_DIR": str(REMOTE_HISTORY_DIR),
    }
    if git_sha is not None:
        env["GPU_KERNELS_GIT_SHA"] = git_sha

    result = _run(command, env=env)
    result["gpu_info"] = gpu_info
    result["history"] = _collect_history(gpu_info)
    return result


@app.local_entrypoint()
def gpu_info(gpu: str = DEFAULT_GPU) -> None:
    """Print GPU and roofline-profile discovery from a Modal GPU."""
    gpu = _validate_gpu(gpu)
    result = _remote(_gpu_info, gpu, gpu)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.local_entrypoint()
def correctness(
    gpu: str = DEFAULT_GPU,
    op: str = "",
    pytest_args: str = "",
) -> None:
    """Run correctness tests on Modal using the same pytest entrypoint as local dev."""
    gpu = _validate_gpu(gpu)
    result = _remote(_run_correctness, gpu, gpu, _correctness_pytest_args(op, pytest_args))
    _print_command_result(result)
    if int(result["returncode"]) != 0:
        raise SystemExit(int(result["returncode"]))


@app.local_entrypoint()
def benchmark(gpu: str = DEFAULT_GPU, op: str = "add", suite_args: str = "") -> None:
    """Run one benchmark suite on Modal and copy its history JSON back locally."""
    gpu = _validate_gpu(gpu)
    ops = BENCHMARK_OPS if op == "all" else (op,)
    unknown = sorted(set(ops) - set(BENCHMARK_OPS))
    if unknown:
        raise SystemExit(f"unknown benchmark op(s): {unknown}; choices: {sorted(BENCHMARK_OPS)}")

    exit_code = 0
    for suite_op in ops:
        print(f"\n### benchmark: {suite_op} on {gpu}")
        result = _remote(
            _run_benchmark,
            gpu,
            gpu,
            suite_op,
            shlex.split(suite_args),
            _local_git_sha(),
        )
        _print_command_result(result)
        _write_local_history(result.get("history", []))
        exit_code = max(exit_code, int(result["returncode"]))

    if exit_code != 0:
        raise SystemExit(exit_code)


def _remote(function: Any, gpu: str, *args: Any) -> Any:
    if gpu.lower() in {"", "none", "cpu"}:
        return function.remote(*args)
    return function.with_options(gpu=gpu).remote(*args)


def _validate_gpu(gpu: str) -> str:
    try:
        return validate_modal_gpu(gpu)
    except ValueError as e:
        choices = ", ".join(SUPPORTED_MODAL_GPUS)
        raise SystemExit(f"unsupported Modal GPU {gpu!r}; use one of: {choices}") from e


def _correctness_pytest_args(op: str, pytest_args: str) -> list[str]:
    if pytest_args:
        return shlex.split(pytest_args)
    if not op:
        return ["tests/correctness", "-q"]
    if op not in BENCHMARK_OPS:
        raise SystemExit(f"unknown correctness op {op!r}; choices: {sorted(BENCHMARK_OPS)}")
    return [f"tests/correctness/test_{op}.py", "-q"]


def _run(command: list[str], *, env: dict[str, str]) -> dict[str, Any]:
    merged_env = os.environ | env
    completed = subprocess.run(
        command,
        cwd=REMOTE_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _collect_history(gpu_info: dict[str, Any]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not REMOTE_HISTORY_DIR.exists():
        return records
    for path in sorted(REMOTE_HISTORY_DIR.glob("*/*.json")):
        record = json.loads(path.read_text())
        record["modal"] = {
            "requested_gpu": gpu_info.get("requested_gpu"),
            "gpu_info": gpu_info,
        }
        records.append(
            {
                "relative_path": str(path.relative_to(REMOTE_HISTORY_DIR)),
                "text": json.dumps(record, indent=2) + "\n",
            }
        )
    return records


def _write_local_history(records: list[dict[str, str]]) -> None:
    for record in records:
        relative_path = Path(record["relative_path"])
        target = _history.HISTORY_DIR / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(record["text"])
        print(f"  copied {target.relative_to(REPO_ROOT)}")


def _print_command_result(result: dict[str, Any]) -> None:
    command = result.get("command")
    if command:
        print(f"$ {shlex.join(command)}")
    if result.get("stdout"):
        print(result["stdout"], end="" if result["stdout"].endswith("\n") else "\n")
    if result.get("stderr"):
        print(
            result["stderr"],
            end="" if result["stderr"].endswith("\n") else "\n",
            file=sys.stderr,
        )
    gpu_info = result.get("gpu_info")
    if gpu_info:
        profile = gpu_info.get("roofline_profile")
        names = ", ".join(device["name"] for device in gpu_info.get("devices", []))
        print(f"  gpu={names or 'unavailable'}  roofline_profile={profile or 'unknown'}")


def _local_git_sha() -> str | None:
    return _history.git_sha()
