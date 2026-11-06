"""DCU runtime and extension checks for native GEMM."""
from __future__ import annotations

import importlib

import torch


_DCU_ARCHITECTURES = frozenset(("gfx926", "gfx928", "gfx936", "gfx938"))


def _require_hip_runtime():
    if not torch.version.hip:
        raise RuntimeError("DCU operators require a HIP-enabled ROCm/DTK PyTorch build")
    if not torch.cuda.is_available():
        raise RuntimeError("DCU operators require a visible, available DCU device")


def require_dcu(*tensors):
    """Require a supported DCU runtime and compatible tensor devices."""
    _require_hip_runtime()
    device = None
    for tensor in tensors:
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.device.type != "cuda":
            raise RuntimeError(f"DCU operator inputs must be on a DCU device, got {tensor.device}")
        if device is not None and tensor.device != device:
            raise RuntimeError("DCU operator inputs must share one device")
        device = tensor.device

    index = torch.cuda.current_device() if device is None or device.index is None else device.index
    try:
        properties = torch.cuda.get_device_properties(index)
    except (AssertionError, RuntimeError) as exc:
        raise RuntimeError(f"cannot inspect DCU device {index}") from exc
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if architecture not in _DCU_ARCHITECTURES:
        allowed = ", ".join(sorted(_DCU_ARCHITECTURES))
        raise RuntimeError(
            f"unsupported DCU architecture {architecture!r}; supported architectures: {allowed}"
        )
    return torch.device("cuda", index)


def _extension():
    _require_hip_runtime()
    try:
        extension = importlib.import_module("kernels._C")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "build the DUMMA GEMM extension with DPA4_BUILD_HIP=1 "
            "DPA4_ENABLE_DUMMA=1 on a supported DCU"
        ) from exc
    if not hasattr(extension, "gemm"):
        raise RuntimeError(
            "the loaded kernels._C extension has no DUMMA GEMM; rebuild with "
            "DPA4_BUILD_HIP=1 DPA4_ENABLE_DUMMA=1"
        )
    return extension


def native_available():
    try:
        require_dcu()
        _extension()
    except (ImportError, OSError, RuntimeError):
        return False
    return True
