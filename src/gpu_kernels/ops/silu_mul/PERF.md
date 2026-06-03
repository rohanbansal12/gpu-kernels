# SiLU Multiply

## Target

- >=70% speed-of-light @ 256 Mi elements, bf16, H100!

## Correctness

- Passed on `H100!` / NVIDIA H100 80GB HBM3:
  `uv run --extra cloud modal run modal_app.py::correctness --op silu_mul`
  (`tests/correctness/test_silu_mul.py -q`, 10 passed).

## Results

Commands:

```bash
uv run --extra cloud modal run modal_app.py::benchmark --gpu H100! --op silu_mul \
  --suite-args "--dtype bf16 --n 268435456 --warmup 10 --iters 30 --block 1024"
uv run --extra cloud modal run modal_app.py::benchmark --gpu H100! --op silu_mul \
  --suite-args "--dtype bf16 --n 268435456 --warmup 10 --iters 30 --sweep-block 256,512,1024,2048,4096"
```

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| H100! | n=256Mi, bf16, block=1024 | torch | 883.60 | 1822.8 | 1.52 | 54.4% | 54.4% |
| H100! | n=256Mi, bf16, block=1024 | torch_compile | 563.12 | 2860.2 | 2.38 | 85.4% | 85.4% |
| H100! | n=256Mi, bf16, block=1024 | jax_jit | 716.17 | 2248.9 | 1.87 | 67.1% | 67.1% |
| H100! | n=256Mi, bf16, block=256 | triton_b256 | 654.96 | 2459.1 | 2.05 | 73.4% | 73.4% |
| H100! | n=256Mi, bf16, block=512 | triton_b512 | 543.02 | 2966.0 | 2.47 | 88.5% | 88.5% |
| H100! | n=256Mi, bf16, block=1024 | triton_b1024 | 545.02 | 2955.1 | 2.46 | 88.2% | 88.2% |
| H100! | n=256Mi, bf16, block=2048 | triton_b2048 | 546.85 | 2945.3 | 2.45 | 87.9% | 87.9% |
| H100! | n=256Mi, bf16, block=4096 | triton_b4096 | 551.39 | 2921.0 | 2.43 | 87.2% | 87.2% |

## Interpretation

- Bottleneck: memory-bound at the measured arithmetic intensity of 0.833 F/B.
- Triton is fastest at block 512, reaching 88.5% SoL and narrowly beating
  `torch_compile` by about 3.6%.
- Blocks 512-2048 are effectively tied; block 256 is clearly under-sized.
- This op is a good Triton candidate: unlike standalone SiLU, the fused gate
  multiply lets the hand-written kernel edge past compiler baselines.

## Next

- Keep block 512 as the default candidate for now.
- Add shape/broadcast validation if this wrapper should support anything beyond
  same-shaped tensors.
