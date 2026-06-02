# Softmax

## Target

- >=60% speed-of-light @ rows=32768, hidden=8192, bf16, H100!

## Correctness

- Not yet run on `H100!`; command: `uv run --extra cloud modal run modal_app.py::correctness --op softmax`

## Results

Use `uv run python -m benchmarks.history_report softmax --latest` after benchmarking, then
copy the important target-relevant rows here. Do not add smoke runs to this table.

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |

## Interpretation

- Bottleneck: unknown until target-size H100! results are measured.
- Notes: softmax mixes reductions, exponentials, and multiple memory passes; SoL is the main comparison.

## Next

- Run correctness on `H100!`, then run target-size bf16 benchmark with a 2-D
  Triton block-size sweep once the Triton implementation is present.
