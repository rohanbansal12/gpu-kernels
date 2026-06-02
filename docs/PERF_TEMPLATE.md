# <Op name>

## Target

- <X% MFU or BW> @ <input shape>, <dtype>, <GPU profile>

## Correctness

- <passed/not run/failed> on <GPU>; command: `<modal correctness command>`

## Results

Use `uv run python -m benchmarks.history_report <op> --latest` after benchmarking, then
copy the important target-relevant rows here. Do not add smoke runs to this table.

| GPU | Config | Variant | Median us | GB/s | TFLOP/s | BW% | SoL% |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| <H100!> | `<shape/dtype/block>` | `<variant>` | <us> | <GB/s> | <TFLOP/s> | <pct> | <pct> |

## Interpretation

- Bottleneck: <one-line hypothesis grounded in the table>
- Notes: <anything important about timing mode, target mismatch, or odd variance>

## Next

- <one concrete follow-up: target-size run, block sweep, correctness fix, accounting fix, etc.>
