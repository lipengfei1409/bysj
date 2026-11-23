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

All twelve ``(focus width, degree)`` pairs are built, in both directions, because
DeepMD's own test suite walks the whole zoo; a single checkpoint only needs its
own pair, so a quick check compiles the glue plus ``(lmax=3, focus_dim=64)`` --
two minutes instead of twenty-four heavy translation units.

Usage (on a DCU host, from the repo root)::

    python csrc/so2_conv_hip/build.py             # build and register
    python csrc/so2_conv_hip/build.py --check     # fast probe (glue + one pair) + schema
    python csrc/so2_conv_hip/build.py --check --full   # all 24 shards
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
BASE_BUILD_DIR = Path(
    os.environ.get("DPA4_SO2_CONV_HIP_BUILD") or (Path.home() / ".cache" / "dpa4_so2_conv_hip")
)


def _defines() -> list[str]:
    """``-D`` flags for the tile sweep (``DPA4_TILE_<name>=<value>``).

    Upstream's launch policy is a measured optimum for an RTX PRO 6000 / H20 and
    its own comment says the constants must be retuned per part; the header
    accepts ``-DDPA4_TILE_<name>=<value>`` so a sweep needs no source edit.
    """
    raw = os.environ.get("DPA4_SO2_CONV_HIP_DEFINES", "")
    return [f"-D{token}" for token in raw.split() if token]


#: Tile knobs the header accepts, and the constraints its ``static_assert``s enforce.
_TILE_NAMES = {"TM", "WARPS", "PK", "OCC"}


def _validate_defines(defines: list[str]) -> None:
    """Reject a tile set the kernel would refuse, before compiling 24 TUs.

    ``so2_conv_kernel_hip.cuh`` asserts ``T::PK % 4 == 0`` (a panel must cover a
    packed step group) and the staging divides ``kWarp`` by ``T::PK``; a swept
    value that violates either fails every translation unit with the same error,
    which is slow to read and slower to compile.
    """
    values: dict[str, int] = {}
    for token in defines:
        name, _, raw = token[2:].partition("=")
        if not name.startswith("DPA4_TILE_"):
            continue
        knob = name[len("DPA4_TILE_"):]
        if knob not in _TILE_NAMES:
            raise RuntimeError(
                f"unknown tile knob {knob}; the header accepts {sorted(_TILE_NAMES)}"
            )
        try:
            values[knob] = int(raw)
        except ValueError as error:
            raise RuntimeError(f"{token}: value must be an integer") from error
        if values[knob] < 1:
            raise RuntimeError(f"{token}: value must be positive")
    pk = values.get("PK")
    if pk is not None and pk % 4 != 0:
        raise RuntimeError(
            f"DPA4_TILE_PK={pk} cannot compile: the kernel asserts PK % 4 == 0 "
            "(a weight panel must cover the packed step group). Use 4, 8, 16 or 32."
        )
    warps = values.get("WARPS")
    if warps is not None and 32 % warps != 0:
        raise RuntimeError(
            f"DPA4_TILE_WARPS={warps} cannot compile: the staging divides the "
            "32-lane unit by WARPS. Use 1, 2, 4, 8 or 32."
        )


def _build_dir(defines: list[str]) -> Path:
    """Cache directory for one define set; a different tile set is a different binary.

    Driven by the environment rather than only by ``--define``, because the
    model path builds through ``model.so2_conv_hip.enable()``: a sweep that only
    tagged the standalone invocation would leave the run loading the default
    library and measure nothing.
    """
    if not defines:
        return BASE_BUILD_DIR
    tag = "-".join(token[2:].replace("=", "") for token in defines)
    return BASE_BUILD_DIR.with_name(f"{BASE_BUILD_DIR.name}_{tag}")


BUILD_DIR = _build_dir(_defines())


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


def compile_commands(
    *, arch: str | None = None, sources: tuple[str, ...] = SOURCES
) -> list[list[str]]:
    """Return one ``hipcc -c`` command per source, plus the final link."""
    includes, link = _torch_flags()
    includes = includes + _defines()
    arch = arch or _arch()
    commands: list[list[str]] = []
    objects: list[str] = []
    for source in sources:
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


#: The glue plus one pair -- enough to catch a header, signature or template
#: error in a couple of minutes instead of compiling twenty-four heavy
#: translation units for the same message.
FAST_SOURCES = ("so2_conv.hip", "so2_conv_fwd_c64_l3.hip", "so2_conv_bwd_c64_l3.hip")


def _up_to_date(sources: tuple[str, ...] = SOURCES) -> bool:
    library = _library_path()
    if not library.is_file():
        return False
    newest_source = max(
        (HERE / source).stat().st_mtime for source in sources
    )
    for header in HERE.glob("*.cuh"):
        newest_source = max(newest_source, header.stat().st_mtime)
    return library.stat().st_mtime >= newest_source


def _require_hip() -> None:
    import torch

    if torch.version.hip is None:
        raise RuntimeError(
            "the fused SO(2) convolution is a HIP operator: build it on a DCU host"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("no visible DCU")


def build(
    *,
    force: bool = False,
    verbose: bool = True,
    dry_run: bool = False,
    fast: bool = False,
) -> Path:
    """Compile and load the extension; returns the ``.so`` path.

    Loading registers the operators, so after this returns
    ``torch.ops.deepmd.dpa4_so2_conv`` exists and DeepMD's own
    ``op_available()`` reports ``True``.
    """
    import torch

    global BUILD_DIR

    _require_hip()
    # Validated here rather than at import so `enable()` can report it as a
    # status reason instead of an import-time traceback.
    _validate_defines(_defines())
    sources = FAST_SOURCES if fast else SOURCES
    if fast:
        # A separate directory so a quick check never invalidates the full build.
        BUILD_DIR = _build_dir(_defines()).with_name(
            _build_dir(_defines()).name + "_fast"
        )
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    commands = compile_commands(sources=sources)
    if dry_run:
        for command in commands:
            print(shlex.join(command))
        return _library_path()

    if force or not _up_to_date(sources):
        # Twenty-four heavy template instantiations; hipcc is single-process per
        # translation unit, so run a few in parallel.  Default low enough not to
        # swap on a shared host.
        jobs = max(1, int(os.environ.get("DPA4_SO2_CONV_HIP_JOBS", "4")))
        failures: list[tuple[str, int]] = []

        def compile_one(command: list[str]) -> None:
            if verbose:
                print("$ " + shlex.join(command), flush=True)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                # Twenty-four TUs fail identically on a bad tile set; keep the
                # first error tail, which is the one worth reading.
                tail = [line for line in result.stderr.splitlines() if "error:" in line][:3]
                failures.append(
                    (Path(command[2]).name, result.returncode, " | ".join(tail) or "(no error line)")
                )

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(compile_one, commands[:-1]))
        if failures:
            name, code, tail = failures[0]
            more = f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
            raise RuntimeError(f"hipcc failed on {name} (exit {code}){more}: {tail}")

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
    parser.add_argument(
        "--define",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra -D for the tile sweep, e.g. --define DPA4_TILE_PK=2 "
             "(builds into a separate cache dir per define set)",
    )
    parser.add_argument("--check", action="store_true", help="report the schema after loading")
    parser.add_argument(
        "--full",
        action="store_true",
        help="compile all twenty-four shards even with --check (the default for "
             "--check is a two-minute glue+one-pair probe)",
    )
    args = parser.parse_args(argv)
    if args.define:
        os.environ["DPA4_SO2_CONV_HIP_DEFINES"] = " ".join(args.define)
    # Re-derive after the environment update so --define and the environment
    # variable take the same path (the latter is what the model path uses).
    global BUILD_DIR
    BUILD_DIR = _build_dir(_defines())
    print(f"tile defines: {' '.join(_defines()) or '(defaults)'}")
    print(f"build dir: {BUILD_DIR}")

    fast = args.check and not args.full
    try:
        library = build(force=args.clean, dry_run=args.dry_run, fast=fast)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    print(f"library: {library}")
    if fast:
        print(
            "fast check: glue + one (lmax=3, focus_dim=64) pair only; "
            "run without --check, or with --check --full, to build all 24 shards"
        )
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
