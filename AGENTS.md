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

Each op's `PERF.md` uses `docs/PERF_TEMPLATE.md`: `Target`, `Correctness`,
`Results`, `Interpretation`, and `Next`. Keep it compact, but include enough
numbers to compare variants and configs. Add a source/reference section only
when there is an op-specific source worth citing.

## Modal workflow

Supported Modal GPU is intentionally narrow:

- `H100!` — no-upgrade H100 for correctness and benchmark comparison.

Before benchmarking an op remotely, run correctness for that op:

```bash
uv run --extra cloud modal run modal_app.py::correctness --op <op>
```

This runs `tests/correctness/test_<op>.py -q` on `H100!` by default and should
cover all implemented variants for that op, including Triton. Use
`--pytest-args` only for unusual debugging. When the user asks for a correctness
check and it passes, update that op's `PERF.md` Correctness section in the same
turn.

For benchmark runs, pass the op and suite args explicitly:

```bash
uv run --extra cloud modal run modal_app.py::benchmark \
  --gpu H100! \
  --op <op> \
  --suite-args "<suite args>"
```

Benchmark JSON is copied back into `bench_history/<op>/`. After any benchmark,
summarize accumulated records so the user can decide what to keep:

```bash
uv run python -m benchmarks.history_report <op> --latest
uv run python -m benchmarks.history_report <op> --best
```

By default, report benchmark findings to the user in the final response and do
not edit `PERF.md`. Only update the Results table when the user explicitly asks
you to update `PERF.md`, save the results, record the benchmark, or similar.
When updating `PERF.md`, include the command or enough config to reproduce it,
the actual GPU/profile, dtype and shape, the important variants, median us,
GB/s, TFLOP/s when relevant, BW% or MFU%, and SoL%. Do not add smoke runs to the
PERF.md results table. If a run is not target-relevant, mention it only in notes
or leave it in `bench_history/`.

When the user asks for a perf check, run a baseline benchmark plus a small but
meaningful block/tile sweep for the op, then report the findings. Start with
these sweep shapes unless the suite or prior results suggest better values:

- 1-D elementwise ops: `--sweep-block 256,512,1024,2048,4096`
- 2-D row/broadcast/reduction ops: `--sweep-block 1,2,4,8 256,512,1024,2048`

If the user also asks to update `PERF.md`, fill the Results table with the
target-relevant best rows and keep the raw detail in `bench_history/`.

Avoid duplicate Triton rows in `PERF.md`. If a baseline run includes `triton`
at the same block/tile size that appears in a sweep, keep the sweep row
(`triton_b...`) and omit the duplicate baseline `triton` row. Keep Torch,
`torch_compile`, and JAX baseline rows because they answer a different question.

Choose targets using `docs/perf_targets.md`. Prefer hardware and roofline facts
first; add an op-specific literature or production-kernel reference before
raising a target beyond the default heuristic.

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
