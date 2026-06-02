# Scale

## Target

- >=80% memory bandwidth @ 256 Mi elements, bf16, H100!

## Correctness

- Passed on `H100!` (`NVIDIA H100 80GB HBM3`); command: `uv run --extra cloud modal run modal_app.py::correctness --op scale`

## Results

Use `uv run python -m benchmarks.history_report scale --latest` after benchmarking, then
copy the important target-relevant rows here.

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100! | `n=268435456 bf16 block=1024` | torch | 366.80 | 2927.3 | 0.73 | 87.4 | 87.4 |
| H100! | `n=268435456 bf16 block=1024` | torch_compile | 402.78 | 2665.8 | 0.67 | 79.6 | 79.6 |
| H100! | `n=268435456 bf16 block=1024` | jax_jit | 554.05 | 1938.0 | 0.48 | 57.9 | 57.9 |
| H100! | `n=268435456 bf16 block=256` | triton_b256 | 682.78 | 1572.6 | 0.39 | 46.9 | 46.9 |
| H100! | `n=268435456 bf16 block=512` | triton_b512 | 411.46 | 2609.6 | 0.65 | 77.9 | 77.9 |
| H100! | `n=268435456 bf16 block=1024` | triton_b1024 | 402.48 | 2667.8 | 0.67 | 79.6 | 79.6 |
| H100! | `n=268435456 bf16 block=2048` | triton_b2048 | 403.86 | 2658.7 | 0.66 | 79.4 | 79.4 |
| H100! | `n=268435456 bf16 block=4096` | triton_b4096 | 405.82 | 2645.8 | 0.66 | 79.0 | 79.0 |

## Interpretation

- Bottleneck: memory-bound. Torch eager is the current winner at 87.4% BW/SoL.
  Triton block sizes 1024-4096 cluster just below the 80% target in the sweep.
- Conclusion: standalone scale does not currently justify Triton on performance
  grounds; keep the Triton kernel as a memory-bandwidth learning case or as a
  building block for future fusion.
- Notes: the baseline run compares Torch/JAX/Triton at block 1024; the sweep
  rows tune Triton block size only. The baseline Triton row is omitted because
  the sweep table already includes block 1024.

## Next

- Try a narrower block sweep around the current winner, for example
  `--sweep-block 768,1024,1280,1536`, then rerun the full baseline at the best
  block to confirm the Torch-vs-Triton gap.
