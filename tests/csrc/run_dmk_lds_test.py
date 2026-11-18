"""Build and run the DMK LDS management tests.

Usage:
    python tests/csrc/run_dmk_lds_test.py [--host-only]

The host test covers the static/dynamic LDS budget, page layout, mbarrier-style
phase arithmetic, page permutation and the globals contract on a machine
without a DCU. The device test runs the instruction/page megakernel and
validates its HBM output across NoOp, elementwise, GEMM, embedding, tail,
idle-block and re-launch cases, so it requires a DTK 26.04 HIP compiler and a
supported DCU.
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


def compile_and_run(command: list[str], executable: Path) -> int:
    print(f"Building {executable.name}...", flush=True)
    built = subprocess.run(command, check=False)
    if built.returncode:
        return built.returncode
    print(f"Running {executable.name}...", flush=True)
    return subprocess.run([str(executable)], check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="skip the DCU device test",
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

    root = Path(__file__).resolve().parents[2]
    common = [
        hipcc,
        "-O2",
        "-std=c++17",
        f"-I{root / 'csrc' / 'dmk'}",
        *(f"-I{path}" for path in include_paths()),
    ]

    with tempfile.TemporaryDirectory(prefix="dmk-lds-test-") as directory:
        workdir = Path(directory)

        host_executable = workdir / "test_dmk_lds"
        status = compile_and_run(
            [
                *common,
                str(root / "tests" / "csrc" / "test_dmk_lds.cpp"),
                "-o",
                str(host_executable),
            ],
            host_executable,
        )
        if status:
            return status
        if args.host_only:
            return 0

        try:
            arch = select_gpu_arch(torch)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 2

        device_executable = workdir / "test_dmk_lds_megakernel"
        return compile_and_run(
            [
                *common,
                f"--offload-arch={arch}",
                str(root / "tests" / "csrc" / "test_dmk_lds_megakernel.cpp"),
                str(root / "csrc" / "megakernel.cpp"),
                "-o",
                str(device_executable),
            ],
            device_executable,
        )


if __name__ == "__main__":
    raise SystemExit(main())
