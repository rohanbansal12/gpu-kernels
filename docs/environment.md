# Environment

This repo assumes `uv` owns the Python environment.

## Create Or Refresh The Local Environment

Core harness only:

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

If starting from an empty folder in the future, the equivalent bootstrap is:

```bash
uv init --package --python 3.12
uv add numpy
uv add --optional torch torch
uv add --optional jax jax
uv add --optional triton triton
uv add --dev pytest ruff pyright
```

This repo already has `pyproject.toml`, so use `uv sync` day to day.

## Local Laptop

On macOS or a CPU-only laptop, use the `local` extra. Expect Torch/JAX
correctness tests to run. Triton/CUDA tests should skip unless you are on a
CUDA-capable NVIDIA machine.

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
