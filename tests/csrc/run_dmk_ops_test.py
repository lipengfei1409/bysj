"""Build and run the standalone dmk operator tests.

Every operator is compiled into its own executable and launched on its own:
one kernel per operator, no megakernel, no fusion.

Usage:
    python tests/csrc/run_dmk_ops_test.py [--op all|attention|gemm]

Requires a DTK 26.04 HIP compiler and a Torch installation providing ATen
headers (the DMK tensor views reuse them).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SUPPORTED_DUMMA_ARCHES = {"gfx926", "gfx928", "gfx936", "gfx938"}

TESTS = {
    "attention": ("test_dmk_attention.cpp", "test_dmk_attention"),
    "gemm": ("test_dmk_gemm_standalone.cpp", "test_dmk_gemm_standalone"),
}


def select_gpu_arch(torch_module) -> str:
    override = os.environ.get("DMK_GPU_ARCH")
    if override:
        arch_name = override
    else:
        try:
            arch_name = torch_module.cuda.get_device_properties(0).gcnArchName
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "Cannot detect DCU architecture; set DMK_GPU_ARCH=gfx928 "
                "for a K100_AI device"
            ) from error
    match = re.match(r"^(gfx[0-9a-f]+)(?::.*)?$", arch_name)
    arch = match.group(1) if match else None
    if arch not in SUPPORTED_DUMMA_ARCHES:
        raise RuntimeError(
            f"Unsupported architecture {arch_name!r}; "
            f"expected one of {sorted(SUPPORTED_DUMMA_ARCHES)}"
        )
    return arch


def build_and_run(command: list[str], executable: Path) -> int:
    print(f"Building {executable.name}...", flush=True)
    built = subprocess.run(command, check=False)
    if built.returncode:
        return built.returncode
    print(f"Running {executable.name}...", flush=True)
    return subprocess.run([str(executable)], check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--op",
        choices=["all", *TESTS],
        default="all",
        help="which standalone operator to build and run (default: all)",
    )
    args = parser.parse_args()

    hipcc = shutil.which("hipcc")
    if hipcc is None:
        print("hipcc not found; source the DTK environment first", file=sys.stderr)
        return 2
    try:
        import torch
        from torch.utils.cpp_extension import include_paths
    except ImportError as error:
        print(f"Torch headers unavailable: {error}", file=sys.stderr)
        return 2

    try:
        arch = select_gpu_arch(torch)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    common = [
        hipcc,
        "-O2",
        "-std=c++17",
        f"--offload-arch={arch}",
        f"-I{root / 'csrc' / 'dmk'}",
        *(f"-I{path}" for path in include_paths()),
    ]

    selected = list(TESTS) if args.op == "all" else [args.op]
    with tempfile.TemporaryDirectory(prefix="dmk-ops-test-") as directory:
        workdir = Path(directory)
        for name in selected:
            source, executable_name = TESTS[name]
            executable = workdir / executable_name
            status = build_and_run(
                [
                    *common,
                    str(root / "tests" / "csrc" / source),
                    "-o",
                    str(executable),
                ],
                executable,
            )
            if status:
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
