#!/usr/bin/env python3
"""Build the translated HIP fused SO(2) convolution as a loadable extension.

Why this does not use ``torch.utils.cpp_extension.load``: on ROCm, ``load``
hipifies every source it is given and takes ``hipify_result[s].hipified_path``,
which is ``None`` for a source hipify did not translate -- and a ``.hip`` source
trips exactly that, so the object list ends up holding ``None`` and ninja's setup
dies with ``basename(None)``.  Since the sources here are already HIP (with four
hand patches hipify cannot make), ``load``'s hipify pass is pointless *and*
breaks the build, so this script drives ``hipcc`` itself and then
``torch.ops.load_library``.

The translated sources register ``deepmd::dpa4_so2_conv`` (and backward,
``dpa4_wigner_runs``, ``dpa4_fp32_ridge``) into the *same* ``deepmd`` operator
namespace DeepMD uses, so upstream's Python wrapper -- ``op_available()``,
``_setup_context``, ``_backward``, the fake impls, the profitability guard --
finds them with no source edit on either side.

v1 ships both directions of one instantiation, ``(lmax=3, focus_dim=64)``, which
is what the profiled checkpoint needs.  The other eleven ``(CF, L)`` pairs are
the same template -- add their shards to ``tools/hipify_so2_conv.py`` when a
checkpoint needs one.

Usage (on a DCU host, from the repo root)::

    python csrc/so2_conv_hip/build.py             # build and register
    python csrc/so2_conv_hip/build.py --check     # build, then report the schema
    python csrc/so2_conv_hip/build.py --dry-run   # print the compile lines only
    python csrc/so2_conv_hip/build.py --clean     # rebuild from scratch
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "dpa4_so2_conv_hip"
#: The glue plus one translation unit per (direction, degree, focus width).
#: All twenty-four are needed to run DeepMD's own test suite, which walks the
#: whole zoo; a single checkpoint only needs its own pair.
SOURCES = ("so2_conv.hip",) + tuple(
    f"so2_conv_{kind}_c{cf}_l{lmax}.hip"
    for cf in (32, 64)
    for lmax in range(1, 7)
    for kind in ("fwd", "bwd")
)
BUILD_DIR = Path(
    os.environ.get("DPA4_SO2_CONV_HIP_BUILD") or (Path.home() / ".cache" / "dpa4_so2_conv_hip")
)


def _compiler() -> str:
    return os.environ.get("HIPCXX") or "hipcc"


def _arch() -> str:
    """Return the offload arch of the visible device, e.g. ``gfx928``."""
    import torch

    name = torch.cuda.get_device_properties(0).gcnArchName
    # The name carries feature suffixes (``gfx928:sramecc+:xnack-``); hipcc
    # wants the bare architecture.
    return name.split(":", 1)[0]


def _torch_flags() -> tuple[list[str], list[str]]:
    """Return ``(compile_flags, link_flags)`` taken from this torch build."""
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    abi = "-D_GLIBCXX_USE_CXX11_ABI=" + ("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0")
    includes = [abi, "-D__HIP_PLATFORM_AMD__"]
    includes += [f"-I{path}" for path in include_paths(device_type="cuda")]
    includes += [f"-I{HERE}"]

    lib_dirs = library_paths(device_type="cuda")
    # ``load`` computes the same list; torch ships libtorch.so as a linker
    # script over the device libraries here.
    link = [f"-L{path}" for path in lib_dirs]
    link += ["-ltorch", "-ltorch_hip", "-lc10"]
    link += [f"-Wl,-rpath,{path}" for path in lib_dirs]
    return includes, link


def _run(command: list[str], verbose: bool) -> int:
    if verbose:
        print("$ " + shlex.join(command), flush=True)
    return subprocess.call(command)


def compile_commands(*, arch: str | None = None) -> list[list[str]]:
    """Return one ``hipcc -c`` command per source, plus the final link."""
    includes, link = _torch_flags()
    arch = arch or _arch()
    commands: list[list[str]] = []
    objects: list[str] = []
    for source in SOURCES:
        obj = str(BUILD_DIR / (Path(source).stem + ".o"))
        objects.append(obj)
        commands.append(
            [
                _compiler(),
                "-c",
                str(HERE / source),
                "-o",
                obj,
                "-O3",
                "-std=c++17",
                "-fPIC",
                "-fno-gpu-rdc",
                f"--offload-arch={arch}",
                *includes,
            ]
        )
    commands.append(
        [
            _compiler(),
            "-shared",
            "-fPIC",
            "-o",
            str(BUILD_DIR / f"{NAME}.so"),
            *objects,
            *link,
        ]
    )
    return commands


def _library_path() -> Path:
    return BUILD_DIR / f"{NAME}.so"


def _up_to_date() -> bool:
    library = _library_path()
    if not library.is_file():
        return False
    newest_source = max(
        (HERE / source).stat().st_mtime for source in SOURCES
    )
    for header in HERE.glob("*.cuh"):
        newest_source = max(newest_source, header.stat().st_mtime)
    return library.stat().st_mtime >= newest_source


def build(*, force: bool = False, verbose: bool = True, dry_run: bool = False) -> Path:
    """Compile and load the extension; returns the ``.so`` path.

    Loading registers the operators, so after this returns
    ``torch.ops.deepmd.dpa4_so2_conv`` exists and DeepMD's own
    ``op_available()`` reports ``True``.
    """
    import torch

    if torch.version.hip is None:
        raise RuntimeError(
            "the fused SO(2) convolution is a HIP operator: build it on a DCU host"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("no visible DCU")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    commands = compile_commands()
    if dry_run:
        for command in commands:
            print(shlex.join(command))
        return _library_path()

    if force or not _up_to_date():
        # Twenty-four heavy template instantiations; hipcc is single-process per
        # translation unit, so run a few in parallel.  Default low enough not to
        # swap on a shared host.
        jobs = max(1, int(os.environ.get("DPA4_SO2_CONV_HIP_JOBS", "4")))
        failures: list[tuple[str, int]] = []

        def compile_one(command: list[str]) -> None:
            code = _run(command, verbose)
            if code != 0:
                failures.append((Path(command[2]).name, code))

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(compile_one, commands[:-1]))
        if failures:
            names = ", ".join(f"{name} (exit {code})" for name, code in failures)
            raise RuntimeError(f"hipcc failed on {names}")

        code = _run(commands[-1], verbose)
        if code != 0:
            raise RuntimeError(f"hipcc failed on the link step (exit {code})")

    torch.ops.load_library(str(_library_path()))
    return _library_path()


def registered() -> bool:
    """Whether ``deepmd::dpa4_so2_conv`` is registered in this process."""
    import torch

    op = getattr(torch.ops.deepmd, "dpa4_so2_conv", None)
    return isinstance(op, torch._ops.OpOverloadPacket)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", help="print the compile lines only")
    parser.add_argument("--clean", action="store_true", help="rebuild every object")
    parser.add_argument("--check", action="store_true", help="report the schema after loading")
    args = parser.parse_args(argv)

    try:
        library = build(force=args.clean, dry_run=args.dry_run)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    print(f"library: {library}")
    print(f"registered: {registered()}")
    if args.check:
        import torch

        for name in (
            "dpa4_so2_conv",
            "dpa4_so2_conv_backward",
            "dpa4_wigner_runs",
            "dpa4_wigner_runs_backward",
            "dpa4_fp32_ridge",
        ):
            packet = getattr(torch.ops.deepmd, name, None)
            print(f"  deepmd::{name}: {'ok' if packet is not None else 'MISSING'}")
        if registered():
            print("\ndefault schema:\n ", torch.ops.deepmd.dpa4_so2_conv.default._schema)
        # DeepMD's guard compares this against 73.2 (the calibration part's
        # ridge); printing it is the honest way to see why the fused path is
        # declined or accepted on this device.
        try:
            print("dpa4_fp32_ridge():", torch.ops.deepmd.dpa4_fp32_ridge())
        except Exception as error:  # noqa: BLE001 - reporting only
            print("dpa4_fp32_ridge() failed:", error)
    return 0


if __name__ == "__main__":
    sys.exit(main())
