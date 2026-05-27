# Scale

Target: >=80% memory bandwidth @ 256 Mi elements, bf16, selected GPU profile
Current: not measured yet (kernel variants are scaffolded only)
Bottleneck: unknown until first Torch/Triton implementations are measured
Next: implement Torch eager, torch.compile, JAX jit, and Triton variants.
