"""SiLU op variants."""

from gpu_kernels.ops.silu.jax_impl import silu_jax, silu_jax_jit
from gpu_kernels.ops.silu.naive import silu_naive
from gpu_kernels.ops.silu.torch_impl import silu_torch, silu_torch_compile
from gpu_kernels.ops.silu.triton_impl import silu_triton

__all__ = [
    "silu_jax",
    "silu_jax_jit",
    "silu_naive",
    "silu_torch",
    "silu_torch_compile",
    "silu_triton",
]
