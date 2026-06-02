# Add

## Target

- >=80% memory bandwidth @ 256 Mi elements, bf16, H100!

## Correctness

- Not yet run on `H100!`; command: `uv run --extra cloud modal run modal_app.py::correctness --op add`

## Results

Use `uv run python -m benchmarks.history_report add --latest` after benchmarking, then
copy the important target-relevant rows here. Do not add smoke runs to this table.

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |

## Interpretation

- Bottleneck: unknown until target-size H100! results are measured.
- Notes: add is expected to be memory-bound for large contiguous inputs.

## Next

- Run correctness on `H100!`, then run target-size bf16 benchmark with a Triton
  block-size sweep.
