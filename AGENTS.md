# AGENTS.md

Agent-facing notes for this repo.

## What this project is

Torch/JAX/Triton GPU kernels with a benchmarking harness built around roofline
analysis. The project is a learning template: correctness first, compiler
baseline second, hand-written Triton only when useful.

## Structure

- `src/gpu_kernels/ops/<op>/` contains op implementations and `PERF.md`.
- `benchmarks/` contains reusable harness code.
- `benchmarks/suites/<op>.py` contains per-op benchmark CLIs.
- `tests/correctness/` compares optimized variants against the readable oracle.
- `bench_history/<op>/` stores benchmark JSON records.

## Per-op contract

Prefer these variants:

1. `naive.py` — readable oracle, not tuned.
2. `torch_impl.py` — eager Torch and optionally `torch.compile`.
3. `jax_impl.py` — optional `jax.jit` comparison.
4. `triton_impl.py` — hand-written Triton when it is justified.

Each op's `PERF.md` has four lines: `Target`, `Current`, `Bottleneck`, `Next`.
Use `docs/PERF_TEMPLATE.md`.

## Benchmark rules

Bench inputs should be random, not all zeros or ones. Count FLOPs and bytes in
the suite before interpreting performance. If speed-of-light exceeds 100%, fix
the accounting, timing, or hardware profile before trusting the run.

## Quality

Run before committing:

```bash
uv run ruff check . && uv run ruff format . && uv run pyright
uv run pytest
```
