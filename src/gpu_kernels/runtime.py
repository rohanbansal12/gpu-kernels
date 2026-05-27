"""Runtime helpers for optional Torch/JAX/Triton environments.

The repo should be useful in three modes:

* core harness only, with no GPU libraries installed;
* local development with Torch/JAX on CPU;
* real benchmarks on a remote NVIDIA GPU with Torch/JAX/Triton + CUDA.

This module keeps imports lazy so the first mode still type-checks and tests.
"""

from __future__ import annotations

from importlib import import_module, util
from typing import Any, Literal

DTypeName = Literal["bf16", "fp16", "fp32", "tf32"]
FrameworkName = Literal["torch", "jax", "triton"]


def has_package(name: str) -> bool:
    """Return whether ``name`` can be imported without importing it."""
    return util.find_spec(name) is not None


def has_torch() -> bool:
    return has_package("torch")


def has_jax() -> bool:
    return has_package("jax")


def has_triton() -> bool:
    return has_package("triton")


def require_torch() -> Any:
    """Import Torch or raise an actionable dependency error."""
    try:
        return import_module("torch")
    except ImportError as e:
        raise RuntimeError(
            "Torch is not installed. Run `uv sync --extra local --group dev`."
        ) from e


def require_jax() -> Any:
    """Import JAX or raise an actionable dependency error."""
    try:
        return import_module("jax")
    except ImportError as e:
        raise RuntimeError("JAX is not installed. Run `uv sync --extra local --group dev`.") from e


def require_jnp() -> Any:
    """Import ``jax.numpy`` or raise an actionable dependency error."""
    try:
        return import_module("jax.numpy")
    except ImportError as e:
        raise RuntimeError("JAX is not installed. Run `uv sync --extra local --group dev`.") from e


def require_triton() -> Any:
    """Import Triton or raise an actionable dependency error."""
    try:
        return import_module("triton")
    except ImportError as e:
        raise RuntimeError(
            "Triton is not installed. Run `uv sync --extra cuda --group dev`."
        ) from e


def torch_cuda_available() -> bool:
    if not has_torch():
        return False
    torch = require_torch()
    return bool(torch.cuda.is_available())


def torch_device(*, prefer_cuda: bool = True) -> Any:
    """Return ``cuda`` when available and requested, otherwise ``cpu``."""
    torch = require_torch()
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def torch_dtype(name: DTypeName) -> Any:
    """Map the repo's dtype names to Torch dtypes."""
    torch = require_torch()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def jax_dtype(name: DTypeName) -> Any:
    """Map the repo's dtype names to JAX dtypes."""
    jnp = require_jnp()
    if name == "bf16":
        return jnp.bfloat16
    if name == "fp16":
        return jnp.float16
    return jnp.float32


def torch_random(
    shape: tuple[int, ...],
    *,
    dtype: DTypeName = "bf16",
    device: Any | None = None,
    seed: int = 0,
) -> Any:
    """Create reproducible Torch random input on CPU or CUDA."""
    torch = require_torch()
    resolved_device = torch_device() if device is None else device
    device_type = getattr(resolved_device, "type", str(resolved_device))
    gen_device = "cuda" if device_type == "cuda" else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    return torch.randn(shape, device=resolved_device, dtype=torch_dtype(dtype), generator=generator)


def jax_random(shape: tuple[int, ...], *, dtype: DTypeName = "bf16", seed: int = 0) -> Any:
    """Create reproducible JAX random input."""
    jax = require_jax()
    return jax.random.normal(jax.random.key(seed), shape, dtype=jax_dtype(dtype))


def torch_to_numpy(value: Any) -> Any:
    """Detach a Torch tensor to NumPy without importing Torch at module import time."""
    return value.detach().cpu().numpy()


def jax_to_numpy(value: Any) -> Any:
    """Convert a JAX array to NumPy."""
    import numpy as np

    return np.asarray(value)


def available_frameworks() -> dict[FrameworkName, bool]:
    return {"torch": has_torch(), "jax": has_jax(), "triton": has_triton()}
