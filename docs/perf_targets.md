# Performance Targets

Targets are working goals for learning and regression tracking, not claims that a
first Triton kernel must reach production-library performance.

## Sources To Use

- Hardware ceilings: use the H100 profile in `benchmarks/roofline.py`, based on
  NVIDIA H100 public specifications for peak compute and HBM bandwidth:
  https://www.nvidia.com/en-us/data-center/h100/
- Model: use roofline analysis: arithmetic intensity, peak compute, and peak
  memory bandwidth determine whether a target should be framed as MFU, BW%, or
  speed-of-light. Overview:
  https://docs.nersc.gov/tools/performance/roofline/
- Operation references: when an op has a known high-quality implementation or
  paper, cite it in notes before raising targets above the default learning
  targets.

## Default Target Heuristics

- Pure elementwise and simple broadcast ops: target memory bandwidth. Start at
  `>=80% BW` for large contiguous inputs.
- Reductions: target memory bandwidth unless the implementation adds enough
  arithmetic or synchronization overhead that speed-of-light is clearer. Start
  at `>=70% BW`.
- Fused activations with transcendental work: target speed-of-light. Start at
  `>=70% SoL`.
- Softmax and normalization: target speed-of-light because they mix reductions,
  elementwise math, and multiple memory passes. Start at `>=60% SoL`.

If a measured run exceeds 100% SoL, treat the target, accounting, timing, or
hardware profile as wrong until proven otherwise.
