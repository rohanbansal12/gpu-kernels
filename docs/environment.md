# Environment

This repo assumes `uv` owns the Python environment.

## Create Or Refresh The Local Environment

Base Torch environment:

```bash
uv sync --group dev
```

macOS/local Torch + JAX development:

```bash
uv sync --extra local --group dev
```

NVIDIA CUDA + Triton development:

```bash
uv sync --extra cuda --group dev
```

Modal remote GPU runs:

```bash
uv sync --extra cloud --group dev
```

If starting from an empty folder in the future, the equivalent bootstrap is:

```bash
uv init --package --python 3.12
uv add numpy
uv add torch
uv add --optional jax jax
uv add --optional triton triton
uv add --optional modal modal
uv add --dev pytest ruff pyright
```

This repo already has `pyproject.toml`, so use `uv sync` day to day.

## Local Laptop

On macOS or a CPU-only laptop, the base environment includes Torch. Use the
`local` extra when you also want JAX comparisons. Triton/CUDA tests should skip
unless you are on a CUDA-capable NVIDIA machine.

```bash
uv sync --extra local --group dev
uv run pytest
uv run ruff check .
uv run pyright
```

## Remote NVIDIA GPU Machine

Use the CUDA extra:

```bash
uv sync --extra cuda --group dev
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

For JAX comparisons, install the CUDA-enabled JAX wheel appropriate for the
remote machine if the plain `jax` dependency does not expose the GPU backend.

## Modal GPUs

`modal_app.py` defines a Modal image with CUDA-capable Torch, JAX, and Triton,
then ships the local `src/`, `benchmarks/`, and `tests/` trees into the remote
container for fast source iteration.

```bash
uv run --extra cloud modal run modal_app.py::gpu_info --gpu H100!
uv run --extra cloud modal run modal_app.py::correctness --op add
uv run --extra cloud modal run modal_app.py::benchmark --gpu H100! --op add
```

Benchmark history is written remotely, enriched with the resolved GPU facts, and
copied back into the local `bench_history/<op>/` directory.
