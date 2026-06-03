# Curriculum

The planned progression for learning GPU kernel work with Torch, JAX, and
Triton. The order is compiler-first: write the readable baseline, measure
Torch/JAX compiler output, then drop to Triton when the custom kernel teaches
something or wins.

## Stage A - Memory And Fusion Basics

### A.1. scale - [complete]

Tiled elementwise hello-world. Teaches benchmark wiring, bytes-vs-FLOPs
accounting, and Triton's program grid. H100 results show Torch eager wins for
standalone scale; Triton is kept as a memory-bandwidth learning kernel and
future fusion building block.

### A.2. add - [complete]

Two-input elementwise add. Teaches multi-input memory accounting:
read `x`, read `y`, write output. H100 results show Torch eager wins for
standalone add; Triton is close but not justified unless fused.

### A.3. bias_add - [complete]

Broadcasted vector add, typically `(M, H) + (H,)`. Teaches broadcast indexing
and the difference between logical shape and memory traffic. H100 results show
Triton narrowly beats compiler baselines for the target shape, but the margin is
small; the main value is broadcast/fusion groundwork.

### A.4. silu - [complete]

Unary activation. Teaches a slightly heavier elementwise body and how math cost
starts to matter even for memory-shaped kernels. H100 results show Torch eager
wins standalone; Triton required bf16-to-fp32 math for sigmoid and remains
useful as an activation/fusion building block.

### A.5. silu_mul - [complete]

SwiGLU-style elementwise fusion: `silu(x) * gate`. Teaches why avoiding an
intermediate activation write can beat composing library calls. H100 results
show Triton block 512 narrowly beats `torch_compile` at the target shape, making
this the first clear fused elementwise Triton win in the curriculum.

### A.6. gelu - [scaffolded]

Common transformer activation with a heavier scalar formula. Good for comparing
Torch eager, `torch.compile`, JAX, and Triton on non-trivial elementwise math.

## Stage B - Reductions

### B.1. row_sum - [scaffolded]

First row-wise reduction. Teaches reduction blocks and accumulation dtype.

### B.2. row_max - [scaffolded]

Same reduction shape as `row_sum`, but prepares the max pass used by softmax.

### B.3. layernorm - [scaffolded]

Mean, variance, normalize, affine. Teaches multi-reduction row kernels.

### B.4. rmsnorm - [scaffolded]

LLM-normalization primitive: mean-square reduction plus rescale. Good
memory-bandwidth target before softmax.

### B.5. softmax - [scaffolded]

Stable row-wise softmax. Teaches max/sum reductions, numerical stability, and
the setup for attention.

## Stage C - Tensor Cores

### C.1. matmul - [planned]

Compare Torch/cuBLAS, JAX/XLA, and Triton matmul. The point is not to beat
cuBLAS first; it is to understand tiling and tensor-core accounting.

### C.2. fused matmul epilogue - [planned]

Matmul plus bias/activation/residual epilogue. Teaches where custom kernels
can beat library calls by avoiding extra memory traffic.

## Stage D - Transformer Kernels

### D.1. RoPE - [planned]

Memory-bound position transform and layout exercise.

### D.2. SwiGLU - [planned]

Elementwise fusion common in LLM MLP blocks.

### D.3. FlashAttention-style attention - [planned]

Online softmax, tiling over sequence, and memory-traffic reduction.

## Stage E - Inference Kernels

### E.1. KV cache append/copy - [planned]

Layout-sensitive serving primitive.

### E.2. quant/dequant - [planned]

Bandwidth-bound low-precision packing and unpacking.

### E.3. MoE dispatch - [planned]

Routing, indexing, and irregular memory movement.
