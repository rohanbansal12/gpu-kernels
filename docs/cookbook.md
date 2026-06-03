# Cookbook

Common commands once suites exist:

```bash
uv sync --extra local --group dev
uv run pytest
uv run ruff check .
uv run pyright
```

Base Torch CPU environment:

```bash
uv sync --group dev
```

NVIDIA CUDA + Triton environment:

```bash
uv sync --extra cuda --group dev
```

Run an op suite:

```bash
uv run python -m benchmarks.suites.<op> --gpu-profile rtx4090
```

Sweep a Triton block axis:

```bash
uv run python -m benchmarks.suites.<op> --sweep-block 256,512,1024
```
