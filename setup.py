"""Build the DPA4 native HIP operators by default."""
import contextlib
import os
import pathlib
import re
import subprocess
import sys

from setuptools import find_packages, setup

# Native kernels use HIP syntax in .cpp translation units. Follow the DTK
# manual's hipcc test.cpp convention rather than dispatching by suffix.
hip_cxx = os.environ.get("HIPCXX") or "hipcc"
os.environ["CXX"] = hip_cxx

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Absolute, because a relative `-Icsrc` is resolved against whatever working
# directory the build tool runs the compiler in -- under torch's ninja backend
# that is the build temp dir, not the project root, so the flag silently points
# at nothing and every `#include "ops/..."` from a subdirectory fails with
# "'ops.h' file not found".
CSRC = pathlib.Path(__file__).resolve().parent / "csrc"

# DTK's du_mma.h supports exactly these four architectures and hard-errors on
# anything else, so a build for another target cannot contain the DUMMA path.
DUMMA_ARCHES = ("gfx926", "gfx928", "gfx936", "gfx938")

#: Where this builds *for*.  gfx928 is the deployment card, so it is the default
#: even when the machine running the build has an older one -- cross-compiling for
#: the target is the normal case here.  An explicit `PYTORCH_ROCM_ARCH` wins.
DEFAULT_ARCH = "gfx928"
EXPLICIT_ARCH = bool(os.environ.get("PYTORCH_ROCM_ARCH"))
ARCH = os.environ.get("PYTORCH_ROCM_ARCH") or DEFAULT_ARCH
os.environ["PYTORCH_ROCM_ARCH"] = ARCH
TARGETS = [a for a in re.split(r"[,;\s]+", ARCH) if a]

def _patch_du_mma(source: str) -> tuple[str, int]:
    """Insert `inline` into the host/device helpers DTK leaves non-inline.

    `du_mma.h` defines `__du_float_to_tf32` as a plain `__host__ __device__` function.
    One translation unit including it is fine, but this build has three kernels, and
    the host link then reports a duplicate symbol.  The definitions are identical, so
    `inline` is the semantics the header should have declared -- we cannot change the
    installed header, so the build compiles against a patched copy that sits earlier
    on the include path.

    Returns the patched text and how many definitions were touched.
    """
    # Line-based on purpose: the header mixes K&R ("signature" then "{" on its own
    # line) with one-liners, and a single regex treats the two differently.
    count = 0
    out: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if (
            stripped.startswith("__host__ __device__")
            and "inline" not in stripped
            and not stripped.rstrip().endswith(";")
        ):
            line = line.replace("__host__ __device__ ",
                                "__host__ __device__ inline ", 1)
            count += 1
        out.append(line)
    return "".join(out), count


def _patched_include_dir() -> str:
    """Directory holding the patched `du_mma.h`, or "" when it cannot be produced."""
    for root in (os.environ.get("HIP_PATH"), os.environ.get("ROCM_PATH"), "/opt/dtk"):
        if not root:
            continue
        candidate = pathlib.Path(root) / "include" / "du_mma.h"
        if not candidate.is_file():
            continue
        try:
            patched, count = _patch_du_mma(candidate.read_text())
        except OSError:
            return ""
        if count == 0:
            return ""  # already inline: the installed header needs no help
        out_dir = pathlib.Path(__file__).resolve().parent / "build" / "dtk-patched"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "du_mma.h").write_text(patched)
        print(f"note: {candidate} defines {count} host/device helper(s) without "
              f"`inline`; compiling against a patched copy in {out_dir}")
        return str(out_dir)
    return ""


PATCHED_INCLUDE = _patched_include_dir()

# `DPA4_GEMM_SCALAR=1` builds the persistent GEMM on the FP32 scalar loop instead
# of DUMMA: the same pages, waves and split, a different inner loop.  It is the way
# to run this on a card with no matrix unit, and the way to separate "is the
# scheduling right" from "is the fragment layout right".
BUILD_SCALAR = os.environ.get("DPA4_GEMM_SCALAR") == "1"
if not BUILD_SCALAR:
    unsupported = [a for a in TARGETS if a not in DUMMA_ARCHES]
    if unsupported:
        raise SystemExit(
            f"PYTORCH_ROCM_ARCH={ARCH} contains {', '.join(unsupported)}, which "
            "DTK's du_mma.h does not support (" + ", ".join(DUMMA_ARCHES) + ").\n"
            "If an old shell exported it, clear it:  unset PYTORCH_ROCM_ARCH\n"
            f"  (the default is {DEFAULT_ARCH}; this build may have been started "
            "with a stale value)\n"
            "or build the scalar GEMM for any architecture:  DPA4_GEMM_SCALAR=1"
        )
CSCALAR_ARGS = ["-DDPA4_GEMM_SCALAR"] if BUILD_SCALAR else []

# `PYTORCH_ROCM_ARCH` alone does not reach hipcc on every torch version: the command
# lines in the build log carried no `--offload-arch` at all, so hipcc fell back to
# its own default target list -- which on DTK includes gfx906, and one failing
# architecture fails the whole translation unit no matter what was asked for.
# Passing the flag here makes the target explicit instead of advisory.
ARCH_FLAGS = [f"--offload-arch={arch}" for arch in TARGETS] if TARGETS else []

# `csrc/gemm.cpp` exposes three build-time switches so one machine can isolate what the
# dense kernel spends its time on without a profiler: padding, k-steps per barrier, and
# a chunked epilogue.  They ride in as -D flags with the file's own defaults.
DENSE_KNOBS = [
    f"-DDPA4_DENSE_{name}={value}"
    for name, value in (
        ("PAD", os.environ.get("DPA4_DENSE_PAD")),
        ("KSTAGE", os.environ.get("DPA4_DENSE_KSTAGE")),
        ("CCHUNK", os.environ.get("DPA4_DENSE_CCHUNK")),
    )
    if value
]


def _arch_from_rocminfo(text: str) -> str:
    """The first agent arch in `rocminfo` output, or "" when there is none."""
    match = re.search(r"amdgcn-amd-amdhsa--(gfx[0-9a-f]+)", text)
    if match is None:
        match = re.search(r"\b(gfx[0-9a-f]+)\b", text)
    return match.group(1) if match else ""


def _device_arch() -> str:
    """The arch of the card in this machine, or "" when it cannot be determined.

    torch is asked first, but its `gcnArchName` has been observed to report an arch
    this build was made for rather than the agent present -- which is how a gfx928
    host came to be treated as gfx906.  `rocminfo` describes the actual agents, so
    it is the fallback and the arbiter when the two disagree.
    """
    try:
        reported = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        reported = reported.strip()
    except Exception:  # noqa: BLE001 - no device visible through torch
        reported = ""
    try:
        listed = _arch_from_rocminfo(
            subprocess.run(["rocminfo"], capture_output=True, text=True,
                           timeout=30, check=False).stdout
        )
    except Exception:  # noqa: BLE001 - rocminfo missing or slow
        listed = ""
    if reported and listed and reported != listed:
        print(f"note: torch reports {reported}, rocminfo reports {listed}; "
              f"using {listed}.")
    return listed or reported


_DEVICE = _device_arch()

if __name__ == "__main__":
    print(f"arch: building for {ARCH}"
          + (" (from PYTORCH_ROCM_ARCH)" if EXPLICIT_ARCH
             else f" (default {DEFAULT_ARCH})")
          + (f"; this device is {_DEVICE}" if _DEVICE else "")
          + ("; inner loop: scalar" if BUILD_SCALAR else "; inner loop: DUMMA")
          + (f"; flags: {' '.join(ARCH_FLAGS)}" if ARCH_FLAGS else "")
          + (f"; dense knobs: {' '.join(DENSE_KNOBS)}" if DENSE_KNOBS else ""))
if _DEVICE and _DEVICE not in TARGETS:
    print(
        f"note: the target ({ARCH}) is not this machine's agent ({_DEVICE}); the "
        "extension will compile for the target and will not run here.  "
        f"\n      unset PYTORCH_ROCM_ARCH   (uses the default {DEFAULT_ARCH})"
        f"\n      or  PYTORCH_ROCM_ARCH={_DEVICE} for this machine."
    )

if not torch.version.hip:
    raise RuntimeError("native build requires a ROCm/DTK PyTorch installation")

sources = [
    # Only the framework for now: `csrc/ops/` is header-only, so the two units
    # below are the persistent megakernel and its Python entry points.  The
    # standalone kernels (rmsnorm, gie, segment_*, force_assembly, slot_ops,
    # edge_combine, monomial, blockdiag, so2_conv) are deliberately out of the
    # build; `csrc/bindings.cpp` is the full registration set for when they
    # return -- swap it in below together with their sources.
    "csrc/bindings_dmk.cpp",
    "csrc/dmk/megakernel.cpp",
    "csrc/dmk/gemm_flat.cpp",
    "csrc/gemm.cpp",
]
ext_modules = [CUDAExtension(
    "kernels._C",
    sources=sources,
    # ops/ and dmk/ are addressed as "ops/..." and "dmk/..." from every
    # translation unit, including the headers they include from each other.
    # The patched DTK header comes first, so `<du_mma.h>` resolves to it.
    include_dirs=[*([PATCHED_INCLUDE] if PATCHED_INCLUDE else []), str(CSRC)],
    define_macros=[],
    # `-Wreturn-type` fires inside torch/pybind headers (830+ lines per build)
    # and drowns the log; the two below are what this tree actually needs to see.
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17", "-Wno-return-type", *ARCH_FLAGS,
                *CSCALAR_ARGS, *DENSE_KNOBS],
    },
)]
#: Where the compiler's full output goes.  A hipcc build of this extension emits
#: hundreds of warnings from torch/pybind headers, and printing them buries the
#: two or three lines that actually matter.
BUILD_LOG = pathlib.Path(__file__).resolve().parent / "build" / "build.log"

#: Errors printed to the terminal before pointing at the log.
MAX_ERRORS_SHOWN = 10


def _fallback_lines(text: str, limit: int = 15) -> list[str]:
    """The tail of a failed log, minus warnings and notes.

    Used when no `error:` line exists -- ninja and the linker do not always use
    that word, and a build that says "subcommand failed" with no error is the one
    case where the last lines are the message.
    """
    keep = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip()
        and "warning:" not in line
        and "note:" not in line
        and not line.startswith("In file included from")
    ]
    return keep[-limit:]


def _error_lines(text: str) -> list[str]:
    """The lines that say something is wrong, without the note/warning noise."""
    patterns = (re.compile(r"\berror:"), re.compile(r"fatal error:"),
                re.compile(r"undefined reference"), re.compile(r"\bFAILED:"))
    seen: list[str] = []
    for line in text.splitlines():
        if any(p.search(line) for p in patterns):
            stripped = line.strip()
            if stripped and stripped not in seen:
                seen.append(stripped)
    return seen


class QuietBuildExt(BuildExtension):
    """Build with the compiler's output in `build/build.log`.

    On failure the log keeps everything, while the terminal gets only the error
    lines and the path -- which is the difference between "scroll for the first
    error" and "read it".
    """

    def build_extensions(self) -> None:
        BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
        # The compiler is a child process and inherits the file descriptors, so a
        # Python-level redirect captures nothing; dup2 moves the descriptors
        # themselves and the whole build lands in the log.
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            with open(BUILD_LOG, "w", encoding="utf-8") as log:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                try:
                    super().build_extensions()
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.dup2(saved_out, 1)
                    os.dup2(saved_err, 2)
                    os.close(saved_out)
                    os.close(saved_err)
        except SystemExit:
            raise
        except BaseException as failure:  # noqa: BLE001 - report, then stop
            # The compiler writes to the redirected file descriptors, so the
            # errors are in the log and *not* in the exception text: scan the log.
            with open(BUILD_LOG, "a", encoding="utf-8") as log:
                log.write(str(failure))
            text = BUILD_LOG.read_text(encoding="utf-8")
            errors = _error_lines(text)
            if errors:
                print(f"\nbuild failed: {len(errors)} error line(s)", file=sys.stderr)
                for line in errors[:MAX_ERRORS_SHOWN]:
                    print(f"  {line}", file=sys.stderr)
                if len(errors) > MAX_ERRORS_SHOWN:
                    print(f"  ... and {len(errors) - MAX_ERRORS_SHOWN} more",
                          file=sys.stderr)
            else:
                print("\nbuild failed: no 'error:' line -- last lines of the log",
                      file=sys.stderr)
                for line in _fallback_lines(text):
                    print(f"  {line}", file=sys.stderr)
            print(f"\nfull output: {BUILD_LOG}", file=sys.stderr)
            raise SystemExit(1) from failure
        else:
            print(f"build ok -- compiler output: {BUILD_LOG}")


cmdclass = {"build_ext": QuietBuildExt}

setup(name="dcu-dpa4", version="0.1.0", python_requires=">=3.9",
      package_dir={"": "python"}, packages=find_packages("python"),
      ext_modules=ext_modules, cmdclass=cmdclass)
