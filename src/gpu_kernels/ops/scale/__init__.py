"""Scale op variants."""

from gpu_kernels.ops.scale.jax_impl import scale_jax, scale_jax_jit
from gpu_kernels.ops.scale.naive import scale_naive
from gpu_kernels.ops.scale.torch_impl import scale_torch, scale_torch_compile
from gpu_kernels.ops.scale.triton_impl import scale_triton

__all__ = [
    "scale_jax",
    "scale_jax_jit",
    "scale_naive",
    "scale_torch",
    "scale_torch_compile",
    "scale_triton",
]
