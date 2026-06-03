# SiLU

## Target

- >=70% speed-of-light @ 256 Mi elements, bf16, H100!

## Correctness

- Passed on `H100!` (`NVIDIA H100 80GB HBM3`); command: `uv run --extra cloud modal run modal_app.py::correctness --op silu`

## Results

Use `uv run python -m benchmarks.history_report silu --latest` after benchmarking, then
copy the important target-relevant rows here. Do not add smoke runs to this table.

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100! | `n=268435456 bf16 block=1024` | torch | 364.43 | 2946.3 | 2.95 | 88.0 | 88.0 |
| H100! | `n=268435456 bf16 block=1024` | torch_compile | 405.87 | 2645.5 | 2.65 | 79.0 | 79.0 |
| H100! | `n=268435456 bf16 block=1024` | jax_jit | 558.05 | 1924.1 | 1.92 | 57.4 | 57.4 |
| H100! | `n=268435456 bf16 block=256` | triton_b256 | 666.05 | 1612.1 | 1.61 | 48.1 | 48.1 |
| H100! | `n=268435456 bf16 block=512` | triton_b512 | 399.46 | 2688.0 | 2.69 | 80.2 | 80.2 |
| H100! | `n=268435456 bf16 block=1024` | triton_b1024 | 384.45 | 2792.9 | 2.79 | 83.4 | 83.4 |
| H100! | `n=268435456 bf16 block=2048` | triton_b2048 | 384.90 | 2789.7 | 2.79 | 83.3 | 83.3 |
| H100! | `n=268435456 bf16 block=4096` | triton_b4096 | 389.68 | 2755.4 | 2.76 | 82.3 | 82.3 |

## Interpretation

- Bottleneck: memory-bound under the current accounting. Torch eager is the
  current winner at 88.0% SoL, with Triton block 1024 close behind at 83.4%.
- Conclusion: standalone SiLU does not currently justify Triton on performance
  grounds, though the Triton kernel is a useful activation/fusion building
  block.
- Notes: Triton SiLU casts bf16 input to fp32 for sigmoid/math before storing
  back to bf16. The baseline Triton row is omitted because the sweep table
  already includes block 1024.

## Next

- Try a narrower sweep around the current Triton winner, for example
  `--sweep-block 768,1024,1280,1536`, then rerun the full baseline at the best
  block to confirm the Torch-vs-Triton gap.
