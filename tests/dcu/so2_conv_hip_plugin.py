"""pytest plugin that loads this project's HIP port of the fused SO(2) conv.

DeepMD's own tests gate on ``torch.ops.deepmd.dpa4_so2_conv`` existing, which is
evaluated at *import* time.  Loading the extension as a pytest plugin happens
before collection, so the operator is registered by the time the gate is read.

Loaded by ``run_upstream_conv_test.py``; not meant to be imported directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

PORT = Path(__file__).resolve().parents[2] / "csrc" / "so2_conv_hip"


def _load() -> None:
    if str(PORT) not in sys.path:
        sys.path.insert(0, str(PORT))
    import build

    if not build.registered():
        build.build()


_load()
