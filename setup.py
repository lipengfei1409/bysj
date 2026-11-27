"""Build the DPA4 native HIP operators by default."""
import contextlib
import os
import pathlib
import re
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

# `DPA4_GEMM_SCALAR=1` builds the persistent GEMM against the FP32 scalar loop
# instead of DUMMA.  It exists because the matrix-unit API in `du_mma.h` differs
# between DTK releases: when the installed header does not expose the fragment
# API, the kernel can still be built, run and measured, and the scheduling side
# (pages, waves, split) can be validated on its own.
CSCALAR_ARGS = ["-DDPA4_GEMM_SCALAR"] if os.environ.get("DPA4_GEMM_SCALAR") == "1" else []

# DTK's du_mma.h hard-errors on anything outside gfx926/gfx928/gfx936/gfx938, and
# torch compiles for every target in PYTORCH_ROCM_ARCH (defaulting to all the
# arches the toolkit knows, gfx906 included).  Fail here, with the fix in the
# message, instead of as "undefined template DUFragment" deep in a device pass.
_DUMMA_ARCHES = ("gfx926", "gfx928", "gfx936", "gfx938")
_ARCH_ENV = os.environ.get("PYTORCH_ROCM_ARCH", "")
if not _ARCH_ENV and not CSCALAR_ARGS:
    # No explicit arch: ask the device.  torch otherwise compiles for every arch
    # the toolkit knows, gfx906 included, and DTK's du_mma.h hard-errors on it.
    try:
        _detected = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001 - no device visible; leave it to the build
        _detected = ""
    if _detected:
        os.environ["PYTORCH_ROCM_ARCH"] = _detected
        _ARCH_ENV = _detected
if not CSCALAR_ARGS and _ARCH_ENV:
    _targets = [a.strip() for a in _ARCH_ENV.replace(";", " ").split(",") if a.strip()]
    _bad = [a for a in _targets if a not in _DUMMA_ARCHES]
    if _bad:
        raise SystemExit(
            "PYTORCH_ROCM_ARCH=" + _ARCH_ENV + " contains "
            + ", ".join(_bad)
            + ", which DTK's du_mma.h does not support ("
            + ", ".join(_DUMMA_ARCHES)
            + ").\nSet PYTORCH_ROCM_ARCH to this card's arch (rocminfo | grep gfx), "
            "or build the scalar GEMM with DPA4_GEMM_SCALAR=1."
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
]
ext_modules = [CUDAExtension(
    "kernels._C",
    sources=sources,
    # ops/ and dmk/ are addressed as "ops/..." and "dmk/..." from every
    # translation unit, including the headers they include from each other.
    include_dirs=[str(CSRC)],
    define_macros=[],
    # `-Wreturn-type` fires inside torch/pybind headers (830+ lines per build)
    # and drowns the log; the two below are what this tree actually needs to see.
    extra_compile_args={"cxx": ["-O3", "-std=c++17", "-Wno-return-type"] + CSCALAR_ARGS},
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
        try:
            with open(BUILD_LOG, "w", encoding="utf-8") as log, \
                    contextlib.redirect_stdout(log), \
                    contextlib.redirect_stderr(log):
                super().build_extensions()
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
