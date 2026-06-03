# Add

## Target

- >=80% memory bandwidth @ 256 Mi elements, bf16, H100!

## Correctness

- Passed on `H100!` (`NVIDIA H100 80GB HBM3`); command: `uv run --extra cloud modal run modal_app.py::correctness --op add`

## Results

Use `uv run python -m benchmarks.history_report add --latest` after benchmarking, then
copy the important target-relevant rows here. Do not add smoke runs to this table.

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100! | `n=268435456 bf16 block=1024` | torch | 525.58 | 3064.4 | 0.51 | 91.5 | 91.5 |
| H100! | `n=268435456 bf16 block=1024` | torch_compile | 551.66 | 2919.6 | 0.49 | 87.2 | 87.2 |
| H100! | `n=268435456 bf16 block=1024` | jax_jit | 657.19 | 2450.8 | 0.41 | 73.2 | 73.2 |
| H100! | `n=268435456 bf16 block=256` | triton_b256 | 663.73 | 2426.6 | 0.40 | 72.4 | 72.4 |
| H100! | `n=268435456 bf16 block=512` | triton_b512 | 546.80 | 2945.5 | 0.49 | 87.9 | 87.9 |
| H100! | `n=268435456 bf16 block=1024` | triton_b1024 | 547.20 | 2943.4 | 0.49 | 87.9 | 87.9 |
| H100! | `n=268435456 bf16 block=2048` | triton_b2048 | 549.07 | 2933.3 | 0.49 | 87.6 | 87.6 |
| H100! | `n=268435456 bf16 block=4096` | triton_b4096 | 550.27 | 2926.9 | 0.49 | 87.4 | 87.4 |

## Interpretation

- Bottleneck: memory-bound. Torch eager is the current winner at 91.5% BW/SoL.
  Triton block sizes 512-4096 cluster around 87-88% BW/SoL; block 256 is too
  small for this workload.
- Conclusion: standalone add does not justify Triton on performance grounds;
  keep the Triton kernel as a memory-bandwidth learning case or as a building
  block for future fusion.
- Notes: the baseline run compares Torch/JAX/Triton at block 1024; the sweep
  rows tune Triton block size only. The baseline Triton row is omitted because
  the sweep table already includes block 1024.

## Next

- Try a narrower block sweep around the current winner, for example
  `--sweep-block 384,512,768,1024`, then rerun the full baseline at the best
  block to confirm the Torch-vs-Triton gap.
