# Curriculum

The planned progression for learning GPU kernel work with Torch, JAX, and
Triton. The order is compiler-first: write the readable baseline, measure
Torch/JAX compiler output, then drop to Triton when the custom kernel teaches
something or wins.

## Stage A - Memory And Fusion Basics

### A.1. scale - [planned]

Tiled elementwise hello-world. Teaches benchmark wiring, bytes-vs-FLOPs
accounting, and Triton's program grid.

### A.2. bias activation - [planned]

Simple fusion: load, add bias, apply activation, store. Teaches when kernel
fusion saves global memory traffic.

### A.3. RMSNorm - [planned]

Row-wise reduction plus elementwise normalize. Teaches reductions, dtype policy,
and a real memory-bandwidth target.

### A.4. softmax - [planned]

Stable row-wise softmax. Teaches max/sum reductions, numerical stability, and
the setup for attention.

## Stage B - Tensor Cores

### B.1. matmul - [planned]

Compare Torch/cuBLAS, JAX/XLA, and Triton matmul. The point is not to beat
cuBLAS first; it is to understand tiling and tensor-core accounting.

### B.2. fused matmul epilogue - [planned]

Matmul plus bias/activation/residual epilogue. Teaches where custom kernels
can beat library calls by avoiding extra memory traffic.

## Stage C - Transformer Kernels

### C.1. RoPE - [planned]

Memory-bound position transform and layout exercise.

### C.2. SwiGLU - [planned]

Elementwise fusion common in LLM MLP blocks.

### C.3. FlashAttention-style attention - [planned]

Online softmax, tiling over sequence, and memory-traffic reduction.

## Stage D - Inference Kernels

### D.1. KV cache append/copy - [planned]

Layout-sensitive serving primitive.

### D.2. quant/dequant - [planned]

Bandwidth-bound low-precision packing and unpacking.

### D.3. MoE dispatch - [planned]

Routing, indexing, and irregular memory movement.
