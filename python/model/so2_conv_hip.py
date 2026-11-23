"""Route DeepMD's fused SO(2) convolution to this project's HIP translation.

DeepMD selects that operator at *model construction* time
(`pt_expt/kernels/cuda/dpa4/so2_conv.py: make_cuda_so2_conv`), behind two gates
this module has to satisfy before the model exists:

1. `DP_CUDA_INFER >= 2` -- read once, when the descriptor is built;
2. `_profitable(conv)` -- upstream's "is the substitution worth it" guard, which
   compares the per-edge mixing cost against the calibration part's ridge point.

The guard is the interesting one.  It answers "does the fused form beat the
*Triton* composition", and on this checkpoint it says no: 225 280 FMA per edge
against a budget of 21 490 at the measured ridge (24.0 vs the calibration part's
73.2).  Our baseline is not Triton though -- Triton cannot compile on this host
at all -- it is the dense path the trace measures (580 ms for the slot segment).
So the substitution is taken over explicitly, and the override is recorded in
`status()` rather than hidden, because it is an engineering judgement and not
upstream's recommendation.

`enable()` must run *before* the model is constructed; `disable()` restores the
guard and the environment variable.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "csrc" / "so2_conv_hip" / "build.py"
GUARD_MODULE = "deepmd.pt_expt.kernels.cuda.dpa4.so2_conv"

_original_profitable = None
_original_cuda_infer = None
_enabled = False


def _load_build_module():
    """Import ``csrc/so2_conv_hip/build.py`` by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("dpa4_so2_conv_hip_build", BUILD)
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"cannot load {BUILD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _guard_module():
    """Return DeepMD's guard module, or ``None`` when DeepMD is not importable."""
    try:
        import deepmd  # noqa: F401
    except ImportError:
        return None
    try:
        return importlib.import_module(GUARD_MODULE)
    except ImportError:
        return None


def enable(*, verbose: bool = False, force: bool = False) -> dict:
    """Build/register the HIP operator and open both gates for the next model.

    Returns a status dict; ``status()["reason"]`` says why it did not take
    effect when it did not.  Calling it twice is a no-op.
    """
    global _original_profitable, _original_cuda_infer, _enabled

    status: dict = {
        "enabled": False,
        "registered": False,
        "guard_overridden": False,
        # Which tile/overlap build this run loaded -- the A/B table needs it to
        # be self-describing, and the defines are what select the binary.
        "defines": os.environ.get("DPA4_SO2_CONV_HIP_DEFINES", ""),
    }
    if _enabled and not force:
        status["enabled"] = True
        status["reason"] = "already enabled"
        return status

    module = _load_build_module()
    try:
        module.build(force=force, verbose=verbose)
    except RuntimeError as error:
        status["reason"] = str(error)
        return status
    status["registered"] = module.registered()
    try:
        import torch

        status["ridge"] = float(torch.ops.deepmd.dpa4_fp32_ridge())
    except Exception as error:  # noqa: BLE001 - reporting only
        status["ridge_error"] = str(error)

    # The gate is read at construction time, so it must be raised before it.
    if _original_cuda_infer is None:
        _original_cuda_infer = os.environ.get("DP_CUDA_INFER")
    current = _original_cuda_infer
    try:
        level = int(current) if current is not None else 0
    except ValueError:
        level = 0
    os.environ["DP_CUDA_INFER"] = str(max(level, 2))
    status["DP_CUDA_INFER"] = os.environ["DP_CUDA_INFER"]

    guard = _guard_module()
    if guard is None:
        status["reason"] = "DeepMD is not importable; nothing to route"
        # Leave the environment variable: harmless, and the caller may still
        # load the extension for a manual check.
        _enabled = status["registered"]
        return status

    if _original_profitable is None:
        _original_profitable = guard._profitable

    def _always_profitable(conv):  # noqa: ANN001 - mirrors upstream's signature
        del conv
        return True

    guard._profitable = _always_profitable
    status["guard_overridden"] = True
    status["enabled"] = True
    if verbose:
        print(f"[so2_conv_hip] DP_CUDA_INFER={os.environ['DP_CUDA_INFER']}, "
              f"ridge={status.get('ridge')}, profitable guard overridden", flush=True)
    _enabled = True
    return status


def disable() -> None:
    """Restore upstream's profitability guard and the environment variable."""
    global _original_profitable, _original_cuda_infer, _enabled

    guard = _guard_module()
    if guard is not None and _original_profitable is not None:
        guard._profitable = _original_profitable
    _original_profitable = None

    if _original_cuda_infer is None:
        os.environ.pop("DP_CUDA_INFER", None)
    else:
        os.environ["DP_CUDA_INFER"] = _original_cuda_infer
    _original_cuda_infer = None
    _enabled = False


def status() -> dict:
    """Current state, for a summary file or a log line."""
    guard = _guard_module()
    return {
        "enabled": _enabled,
        "guard_overridden": bool(
            guard is not None and _original_profitable is not None
            and guard._profitable is not _original_profitable
        ),
        "DP_CUDA_INFER": os.environ.get("DP_CUDA_INFER"),
    }
