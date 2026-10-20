"""Configure eager paths before constructing a DeePMD PyTorch model."""
import os
import sys

import torch


EAGER_FLAGS=("DP_COMPILE_INFER","DP_TRITON_INFER","DP_CUDA_INFER","DP_CUTE_INFER",
    "DP_CUTILE_INFER","DP_ACT_INFER","DP_AMP_INFER")


def configure_eager(device=None):
    if device not in (None,"cpu","dcu"):
        raise ValueError("device must be cpu or dcu")
    if device=="dcu" and not (torch.version.hip and torch.cuda.is_available()):
        raise RuntimeError("DCU request requires ROCm/DTK PyTorch and an available DCU; refusing CPU fallback")
    loaded=sys.modules.get("deepmd.pt.utils.env")
    if loaded is not None and device is not None:
        expected="cpu" if device=="cpu" else "cuda"
        if not str(loaded.DEVICE).startswith(expected):
            raise RuntimeError("DeePMD device already initialized; start a fresh process")
    if device=="cpu":
        os.environ["DEVICE"]="cpu"
    elif device=="dcu":
        os.environ.pop("DEVICE",None)
    for flag in EAGER_FLAGS:
        os.environ[flag]="0"
