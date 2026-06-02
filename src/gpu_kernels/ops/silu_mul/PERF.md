# SiLU Multiply

## Target

- >=70% speed-of-light @ 256 Mi elements, bf16, H100!

## Correctness

- Not yet run on `H100!`; command: `uv run --extra cloud modal run modal_app.py::correctness --op silu_mul`

## Results

Use `uv run python -m benchmarks.history_report silu_mul --latest` after benchmarking, then
copy the important target-relevant rows here. Do not add smoke runs to this table.

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |

## Interpretation

- Bottleneck: unknown until target-size H100! results are measured.
- Notes: SiLU multiply fuses activation and gating, so SoL is clearer than raw bandwidth.

## Next

- Run correctness on `H100!`, then run target-size bf16 benchmark with a Triton
  block-size sweep once the Triton implementation is present.
