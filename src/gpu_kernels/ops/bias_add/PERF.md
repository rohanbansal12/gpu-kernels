# Bias Add

## Target

- >=80% memory bandwidth @ rows=32768, hidden=8192, bf16, H100!

## Correctness

- Passed on `H100!` (`NVIDIA H100 80GB HBM3`); command: `uv run --extra cloud modal run modal_app.py::correctness --op bias_add`

## Results

Use `uv run python -m benchmarks.history_report bias_add --latest` after benchmarking, then
copy the important target-relevant rows here. Do not add smoke runs to this table.

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100! | `rows=32768 hidden=8192 bf16 block=16x1024` | torch | 685.70 | 1565.9 | 0.39 | 46.7 | 46.7 |
| H100! | `rows=32768 hidden=8192 bf16 block=16x1024` | torch_compile | 398.96 | 2691.4 | 0.67 | 80.3 | 80.3 |
| H100! | `rows=32768 hidden=8192 bf16 block=16x1024` | jax_jit | 574.65 | 1868.6 | 0.47 | 55.8 | 55.8 |
| H100! | `rows=32768 hidden=8192 bf16 block=8x2048` | triton_b8x2048 | 394.18 | 2724.1 | 0.68 | 81.3 | 81.3 |
| H100! | `rows=32768 hidden=8192 bf16 block=16x2048` | triton_b16x2048 | 392.26 | 2737.4 | 0.68 | 81.7 | 81.7 |
| H100! | `rows=32768 hidden=8192 bf16 block=32x1024` | triton_b32x1024 | 396.30 | 2709.4 | 0.68 | 80.9 | 80.9 |

## Interpretation

- Bottleneck: memory-bound. Triton is the current winner at 81.7% BW/SoL with
  block 16x2048, narrowly ahead of torch_compile at 80.3%.
- Conclusion: standalone bias_add can justify Triton modestly over compiler
  baselines, but the margin is small; the main value is as a broadcast/fusion
  building block.
- Notes: the baseline run compares Torch/JAX/Triton at block 16x1024; the sweep
  rows tune Triton block size only. The baseline Triton row is omitted because
  the sweep covers nearby/equivalent Triton configs. Sweep config 32x2048 was an
  outlier at 15.1% BW/SoL and should be avoided.

## Next

- Try a narrower sweep around 16x2048, for example
  `--sweep-block 8,16,24 1536,2048,2560`, then rerun the full baseline at the
  best block to confirm the torch_compile-vs-Triton gap.
