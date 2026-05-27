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

More detail lives in [`docs/environment.md`](docs/environment.md).

## Variant Ladder

For each op, prefer this progression:

```text
naive -> torch eager -> torch.compile -> jax.jit -> triton
```

Not every op needs every rung. Triton is justified when it beats the compiler,
exposes an important GPU concept, or enables a fusion/layout the high-level
framework does not express well.
