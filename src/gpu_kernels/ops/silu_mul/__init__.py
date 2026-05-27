"""SiLU-multiply op variants."""

from gpu_kernels.ops.silu_mul.jax_impl import silu_mul_jax, silu_mul_jax_jit
from gpu_kernels.ops.silu_mul.naive import silu_mul_naive
from gpu_kernels.ops.silu_mul.torch_impl import silu_mul_torch, silu_mul_torch_compile
from gpu_kernels.ops.silu_mul.triton_impl import silu_mul_triton

__all__ = [
    "silu_mul_jax",
    "silu_mul_jax_jit",
    "silu_mul_naive",
    "silu_mul_torch",
    "silu_mul_torch_compile",
    "silu_mul_triton",
]
