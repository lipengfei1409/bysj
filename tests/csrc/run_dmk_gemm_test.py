"""Build and run the direct DUMMA GemmOp accuracy/performance test.

Usage:
    python tests/csrc/run_dmk_gemm_test.py [M N K REPEATS]

Defaults to M=N=K=1024 and 30 timed launches. Requires a DTK 26.04
HIP compiler and a Torch installation providing ATen headers.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) not in (1, 5):
        print(f"usage: {sys.argv[0]} [M N K REPEATS]", file=sys.stderr)
        return 2
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        print("hipcc not found; source the DTK environment first", file=sys.stderr)
        return 2
    try:
        from torch.utils.cpp_extension import include_paths
    except ImportError as error:
        print(f"Torch headers unavailable: {error}", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    source = root / "tests" / "csrc" / "test_dmk_gemm_accuracy_perf.cpp"
    with tempfile.TemporaryDirectory(prefix="dmk-gemm-test-") as directory:
        executable = Path(directory) / "test_dmk_gemm_accuracy_perf"
        command = [
            hipcc,
            "-O3",
            "-std=c++17",
            f"-I{root / 'csrc' / 'dmk'}",
            *(f"-I{path}" for path in include_paths()),
            str(source),
            "-o",
            str(executable),
        ]
        print("Building direct DUMMA GEMM test...", flush=True)
        built = subprocess.run(command, check=False)
        if built.returncode:
            return built.returncode
        print("Running accuracy and performance cases...", flush=True)
        return subprocess.run([str(executable), *sys.argv[1:]], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
