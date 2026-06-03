# gpu-kernels

A repo for learning GPU kernel work in the order most useful for modern
AI projects: start with readable Torch/JAX baselines, try compiler-backed
variants such as `torch.compile` / `jax.jit`, then write Triton kernels only
where the compiler leaves performance on the table or the kernel itself is the
lesson.

The project is intentionally shaped like a lab notebook with guardrails:

1. **Correctness oracle first** — every op has a readable implementation that
   optimized variants compare against.
2. **Benchmark against a roofline** — results report FLOP/s, memory bandwidth,
   and speed-of-light percent instead of only milliseconds.
3. **Keep performance history** — benchmark JSON lands in `bench_history/` so
   regressions and wins are tied to the code that produced them.

## Layout

- `src/gpu_kernels/ops/<op>/` — op implementations.
- `benchmarks/` — timing, roofline, comparison, and sweep harness.
- `benchmarks/suites/<op>.py` — one benchmark CLI per op.
- `tests/` — correctness and harness tests.
- `docs/curriculum.md` — planned learning order.
- `docs/kernel_recipes.md` — reusable kernel patterns once proven.

## Environment

This repo uses `uv`.

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

VS Code/Pylance Triton autocomplete on macOS:

```bash
mkdir -p .external
git clone --depth 1 https://github.com/triton-lang/triton .external/triton
```

The workspace setting in `.vscode/settings.json` points Pylance at
`.external/triton/python` for editor hints without installing Triton locally.
Kernel execution still runs through CUDA or Modal.

More detail lives in [`docs/environment.md`](docs/environment.md).

## Variant Ladder

For each op, prefer this progression:

```text
naive -> torch eager -> torch.compile -> jax.jit -> triton
```

Not every op needs every rung. Triton is justified when it beats the compiler,
exposes an important GPU concept, or enables a fusion/layout the high-level
framework does not express well.

## Correctness Commands

For local Torch/JAX correctness on one op, use the op's pytest file:

```bash
uv run --extra local pytest tests/correctness/test_scale.py -q
```

Replace `scale` with the op name. Local CUDA/Triton cases skip unless the
machine has CUDA available through PyTorch.

For remote correctness on the supported Modal GPU:

```bash
uv run --extra cloud modal run modal_app.py::correctness --op scale
```

This runs `tests/correctness/test_scale.py -q` on `H100!`.

## Modal GPU Loop

`modal_app.py` mirrors the local iteration cycle on Modal GPUs:

```bash
uv run --extra cloud modal run modal_app.py::gpu_info --gpu H100!
uv run --extra cloud modal run modal_app.py::correctness --op add
uv run --extra cloud modal run modal_app.py::benchmark --gpu H100! --op add
```

The only supported Modal GPU string is `H100!`. The bang is intentional: it
tells Modal not to upgrade H100 requests to H200, which keeps benchmark
comparisons stable. Benchmark runs infer the roofline profile from the actual
CUDA device and copy generated JSON back into `bench_history/<op>/`.

Pass through existing pytest or suite options as strings:

```bash
uv run --extra cloud modal run modal_app.py::correctness \
  --op add

uv run --extra cloud modal run modal_app.py::benchmark \
  --gpu H100! \
  --op softmax \
  --suite-args "--dtype fp16 --rows 8192 --hidden 4096 --warmup 10 --iters 50"
```

Summarize accumulated runs for one op:

```bash
uv run python -m benchmarks.history_report scale
uv run python -m benchmarks.history_report scale --latest
uv run python -m benchmarks.history_report scale --best
```

Correctness and benchmark runs default to `H100!`. Benchmark runs should name
the suite config explicitly. After benchmarking, use the history report output
to inspect the actual GPU/profile, dtype, shape, important variants, median
time, GB/s, TFLOP/s, BW%/MFU%, and SoL%.

When asking an agent for correctness or perf checks, expect the matching
`src/gpu_kernels/ops/<op>/PERF.md` Correctness section to be updated after a
passing correctness run. Benchmark checks report findings by default; ask
explicitly to update `PERF.md` when you want target-relevant rows written to the
table. Do not add smoke runs to the PERF table.
