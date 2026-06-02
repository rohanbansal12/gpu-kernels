"""GPU discovery helpers shared by local and Modal runners."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

SUPPORTED_MODAL_GPUS = ("H100!",)


@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    name: str
    total_memory_mb: int | None
    capability: str | None


@dataclass(frozen=True, slots=True)
class GpuInfo:
    requested_gpu: str | None
    torch_cuda_available: bool
    torch_version: str | None
    cuda_version: str | None
    device_count: int
    devices: list[GpuDevice]
    nvidia_smi: str | None
    roofline_profile: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILE_NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    ("h100", "h100"),
    ("4090", "rtx4090"),
)

MODAL_GPU_PROFILE_HINTS: tuple[tuple[str, str], ...] = (("h100", "h100"),)


def collect_gpu_info(*, requested_gpu: str | None = None) -> GpuInfo:
    """Collect CUDA facts without making the benchmark runner Modal-specific."""
    torch = _import_optional("torch")
    torch_version = getattr(torch, "__version__", None) if torch is not None else None
    cuda_version = (
        getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None
    )
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    devices: list[GpuDevice] = []
    if cuda_available and torch is not None:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            capability = f"{props.major}.{props.minor}"
            total_memory_mb = int(props.total_memory // (1024 * 1024))
            devices.append(
                GpuDevice(
                    index=index,
                    name=torch.cuda.get_device_name(index),
                    total_memory_mb=total_memory_mb,
                    capability=capability,
                )
            )

    nvidia_smi = _nvidia_smi()
    profile_name = roofline_profile_for(
        requested_gpu=requested_gpu,
        device_names=[device.name for device in devices],
        nvidia_smi=nvidia_smi,
    )
    return GpuInfo(
        requested_gpu=requested_gpu,
        torch_cuda_available=cuda_available,
        torch_version=torch_version,
        cuda_version=cuda_version,
        device_count=len(devices),
        devices=devices,
        nvidia_smi=nvidia_smi,
        roofline_profile=profile_name,
    )


def roofline_profile_for(
    *,
    requested_gpu: str | None,
    device_names: list[str],
    nvidia_smi: str | None = None,
) -> str | None:
    """Map actual or requested GPU names to one of ``benchmarks.roofline.PROFILES``."""
    for device_name in device_names:
        profile_name = profile_name_from_device_name(device_name)
        if profile_name is not None:
            return profile_name

    if nvidia_smi:
        profile_name = profile_name_from_device_name(nvidia_smi)
        if profile_name is not None:
            return profile_name

    return profile_name_from_modal_request(requested_gpu)


def profile_name_from_device_name(name: str) -> str | None:
    """Best-effort roofline profile lookup from CUDA or ``nvidia-smi`` names."""
    normalized = _normalize_name(name)
    for pattern, profile_name in PROFILE_NAME_PATTERNS:
        if pattern in normalized:
            return profile_name
    return None


def profile_name_from_modal_request(requested_gpu: str | None) -> str | None:
    """Best-effort roofline profile lookup from Modal's GPU request string."""
    if not requested_gpu:
        return None
    first_choice = re.split(r"[,;]", requested_gpu)[0].strip()
    base = first_choice.split(":", maxsplit=1)[0].rstrip("!+").lower()
    for pattern, profile_name in MODAL_GPU_PROFILE_HINTS:
        if base == pattern:
            return profile_name
    if base == "any":
        return None
    return None


def validate_modal_gpu(gpu: str) -> str:
    """Validate the selected Modal GPU lane before starting remote work."""
    normalized = _normalize_gpu_request(gpu)
    allowed = {_normalize_gpu_request(item) for item in SUPPORTED_MODAL_GPUS}
    requested = normalized.split(":", maxsplit=1)[0]
    if requested not in allowed:
        choices = ", ".join(SUPPORTED_MODAL_GPUS)
        raise ValueError(f"unsupported Modal GPU {gpu!r}; use one of: {choices}")
    return gpu


def format_gpu_info(info: GpuInfo) -> str:
    """Stable JSON formatting for CLI output."""
    return json.dumps(info.to_dict(), indent=2, sort_keys=True)


def _import_optional(name: str) -> Any | None:
    try:
        return import_module(name)
    except ImportError:
        return None


def _nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _normalize_gpu_request(gpu: str) -> str:
    return gpu.strip().lower()
