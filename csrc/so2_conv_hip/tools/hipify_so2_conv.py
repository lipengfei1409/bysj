#!/usr/bin/env python3
"""Translate DeepMD's CUDA fused SO(2) convolution into a HIP source tree.

This is a build-time tool, run on any machine (it only needs ``torch`` for
``torch.utils.hipify``); the output is checked in so the port is auditable and
the translation is reproducible:

    python csrc/so2_conv_hip/tools/hipify_so2_conv.py --deepmd-kit ../deepmd-kit

Pipeline per file: copy -> ``torch.utils.hipify.hipify`` -> local patches.  The
patches carry everything hipify cannot know:

1. **The 32-lane unit is deliberate.** The CUDA kernel is written for a 32-lane
   warp: every output column is ``j * 32 + lane`` and ``kWarp = 32`` splits the
   channel width.  A ROCm wavefront is 64 lanes, so the port keeps the 32-lane
   unit and runs two of them per wave; ``__shfl*_sync`` therefore becomes
   ``__shfl*(v, d, kWarp)``, which shuffles inside each 32-lane segment.
2. **``__syncwarp()`` sits inside divergent code**, so it must become a
   wave-level barrier, not ``__syncthreads()`` (which would deadlock).
3. **The async pipeline becomes synchronous.** ``__pipeline_memcpy_async`` on a
   4-float vector is a plain device copy and the two waits collapse; the
   ``__syncthreads()`` on both sides already orders the shared buffer.  v1
   trades the double-buffering away for correctness; the comment marks the
   place to reintroduce it with ``hip::memcpy_async``.
4. **``dpa4_fp32_ridge`` is rewritten**: it read CUDA compute capability and a
   128-lane FMA width.  On ROCm the FMA width is the wavefront size and the
   clocks come from ``hipDeviceProp_t``.
5. **No ATen masquerading headers, and no torch device guard.** hipify turns
   ``ATen/cuda/CUDAContext.h`` into ``ATen/hip/HIPContext.h``, whose
   ``ATen/hip/Exceptions.h`` does ``#include <cusolver_common.h>`` -- a header
   no ROCm install ships.  The current stream comes from ``c10/hip/HIPStream.h``
   instead, and the device guard is a local RAII wrapper over
   ``hipGetDevice``/``hipSetDevice``: the c10/vendor guard header lives in a
   different place in every torch build (``c10/hip/impl/`` here, absent there),
   while the plain HIP calls are always present.

Exit code is non-zero when a CUDA-only token survives, so this doubles as the
lint the host test runs.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent

#: Files the extension needs.  The 12 forward/backward shards instantiate one
#: ``(CF, L)`` pair each; v1 ships ``c64_l3`` only, which is what this
#: checkpoint needs, and the others can be added by copying the template.
#: Every ``(focus width, degree)`` pair upstream instantiates, both directions.
#: All twelve are built because DeepMD's own test suite exercises the whole zoo
#: (nano=L1/C32 .. pro=L5/C64) and any missing pair turns a numerical test into
#: a "no instantiation" error.  Building one pair is enough to *run* a specific
#: checkpoint, but not to validate the translation.
_PAIRS = tuple((lmax, cf) for cf in (32, 64) for lmax in range(1, 7))
SHARDS = tuple(
    (kind, lmax, cf) for kind in ("fwd", "bwd") for lmax, cf in _PAIRS
)
FILES = (
    "so2_conv.cu",
    "so2_conv.cuh",
    "so2_conv_kernel.cuh",
    "so2_conv_launch.h",
    "so2_conv_instantiate.cuh",
) + tuple(f"so2_conv_{kind}_c{cf}_l{lmax}.cu" for kind, lmax, cf in SHARDS)

#: ``(lmax, focus width)`` pairs this build instantiates, derived from SHARDS.
SHAPES = tuple(sorted({(lmax, cf) for _kind, lmax, cf in SHARDS}))

#: Tokens that must not survive into the HIP tree.
FORBIDDEN = (
    "cudaStream_t",
    "cudaGetLastError",
    "cudaSuccess",
    "cudaFuncSetAttribute",
    "cudaDevAttr",
    "cuda_runtime.h",
    "cuda_pipeline",
    "__pipeline_",
    "__shfl_sync",
    "__shfl_xor_sync",
)

#: hipify's ``<<<>>>`` rewrite nests ``dim3(dim3(x))``; harmless but noisy.
_DIM3 = re.compile(r"dim3\(dim3\(([^()]*(?:\([^()]*\))?[^()]*)\)\)")

#: hipify's ATen masquerading header pulls ``cusolver_common.h``, which no ROCm
#: install ships.  The stream still comes from torch (c10 has it everywhere);
#: the device guard does not, because the guard header's location differs across
#: torch builds.
_INCLUDE_MAP = {
    "#include <ATen/hip/HIPContext.h>": "#include <c10/hip/HIPStream.h>",
    "#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n": "",
}
_API_MAP = {
    "at::hip::getCurrentHIPStreamMasqueradingAsCUDA()": "c10::hip::getCurrentHIPStream()",
    "at::hip::OptionalHIPGuardMasqueradingAsCUDA": "dpa4::DeviceGuard",
}

_DEVICE_GUARD = """
namespace dpa4 {

/// RAII device guard over the plain HIP API.
///
/// torch ships a guard header, but its path differs between builds
/// (``c10/hip/impl/``, ``ATen/hip/impl/``, or absent), and pulling ATen's
/// masquerading headers drags in ``cusolver_common.h``.  ``hipGetDevice`` and
/// ``hipSetDevice`` are always available and are all the operator needs: torch
/// dispatch has normally selected the device already, so this is belt and
/// braces around a kernel launch.
class DeviceGuard {
 public:
  explicit DeviceGuard(const c10::Device& device) : previous_(-1) {
    hipGetDevice(&previous_);
    if (device.has_index() && device.index() >= 0) {
      hipSetDevice(static_cast<int>(device.index()));
    }
  }
  ~DeviceGuard() {
    if (previous_ >= 0) {
      hipSetDevice(previous_);
    }
  }
  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int previous_;
};

}  // namespace dpa4
"""

_SHAPE_MACRO = """
// Only the shapes this build instantiates.  Upstream's
// ``DPA4_CONV_FOR_EACH_SHAPE`` lists all twelve (focus width, degree) pairs and
// the dispatch below is expanded over it, which references eleven
// ``conv_*_launch`` symbols nothing here defines -- the link succeeds but
// ``load_library`` fails on the first undefined symbol.  Generated from the
// shard list in tools/hipify_so2_conv.py.
#define DPA4_CONV_SHIPPED_SHAPES(macro) {shapes}
"""


def _shape_macro() -> str:
    return _SHAPE_MACRO.format(
        shapes=" ".join(f"macro({lmax}, {cf})" for lmax, cf in SHAPES)
    )


def _restrict_shapes(text: str) -> str:
    """Point the dispatch and the validation at the instantiated shapes only."""
    text = text.replace(
        "DPA4_CONV_FOR_EACH_SHAPE(DPA4_CASE)", "DPA4_CONV_SHIPPED_SHAPES(DPA4_CASE)"
    )
    text = text.replace(
        "\n#include <cmath>", "\n" + _shape_macro() + "\n\n#include <cmath>", 1
    )
    start = text.index("bool conv_shape_instantiated(int lmax, int focus_dim) {")
    end = text.index("\n}\n", start) + 3
    body = """bool conv_shape_instantiated(int lmax, int focus_dim) {
  // Generated from the compiled shards; upstream's version answers for all
  // twelve, which would let a shape through validation and then fail the link.
  bool instantiated = false;
#define DPA4_INSTANTIATED_CASE(LV, CFV) \
  if (lmax == LV && focus_dim == CFV) { \
    instantiated = true;                \
  }
  DPA4_CONV_SHIPPED_SHAPES(DPA4_INSTANTIATED_CASE)
#undef DPA4_INSTANTIATED_CASE
  return instantiated;
}
"""
    return text[:start] + body + text[end:]

_RIDGE = '''double dpa4_fp32_ridge() {
  int dev = 0;
  hipGetDevice(&dev);
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    return 0.0;
  }
  // fp32 FMA lanes per compute unit on CDNA/RDNA is the wavefront size; the
  // arithmetic below is the same roofline ratio the CUDA build computes.
  const double lanes = static_cast<double>(prop.warpSize);
  const double flops =
      2.0 * lanes * prop.multiProcessorCount * (prop.clockRate * 1e3);
  const double bytes =
      (prop.memoryClockRate * 1e3) * (prop.memoryBusWidth / 8.0) * 2.0;
  return bytes > 0.0 ? flops / bytes : 0.0;
}'''


def _hipify(src: Path, work: Path) -> dict[str, Path]:
    from torch.utils.hipify import hipify_python

    if work.exists():
        shutil.rmtree(work)
    (work / "in").mkdir(parents=True)
    # hipify copies the input tree itself and only then processes it; a
    # pre-created (empty) output directory makes it find no files at all.
    for name in FILES:
        shutil.copy2(src / name, work / "in" / name)
    hipify_python.hipify(
        project_directory=str(work / "in"),
        output_directory=str(work / "out"),
        extensions=(".cu", ".cuh", ".h"),
        is_pytorch_extension=True,
        show_progress=False,
    )
    return {path.name: path for path in (work / "out").iterdir()}


def _patch(text: str, *, stamp: str) -> str:
    # 4-float pipeline copy -> a plain device copy; the barriers around the
    # staging loop already order the shared buffer.
    text = text.replace(
        "__pipeline_memcpy_async(into + at, from + at, VEC * sizeof(float));",
        "// v1: synchronous stage; reintroduce hip::memcpy_async for double\n"
        "      // buffering once the port is proven.\n"
        "      __builtin_memcpy(into + at, from + at, VEC * sizeof(float));",
    )
    text = text.replace("__pipeline_commit();", "// commit: synchronous stage")
    text = text.replace("__pipeline_wait_prior(1);", "// wait: synchronous stage")
    text = text.replace("__pipeline_wait_prior(0);", "// wait: synchronous stage")
    # The 32-lane unit is preserved on purpose (see the module docstring).
    text = text.replace(
        "__shfl_xor_sync(kFullMask, v, d)", "__shfl_xor(v, d, kWarp)"
    )
    text = text.replace("__shfl_sync(kFullMask, ", "__shfl(")
    text = text.replace("__syncwarp();", "dpa4_wave_barrier();")
    text = _DIM3.sub(r"dim3(\1)", text)
    text = re.sub(r"#include <cuda_pipeline\.h>\n", "", text)
    for old, new in _INCLUDE_MAP.items():
        text = text.replace(old, new)
    for old, new in _API_MAP.items():
        text = text.replace(old, new)
    if "class DeviceGuard" not in text:
        text = text.replace(
            "#include <torch/torch.h>", "#include <torch/torch.h>\n" + _DEVICE_GUARD, 1
        )
    # Every translation unit needs the shims before its first device function.
    if stamp not in text:
        text = text.replace(
            "#include <hip/hip_runtime.h>",
            '#include <hip/hip_runtime.h>\n\n#include "hip_compat.cuh"',
            1,
        )
        if '"hip_compat.cuh"' not in text:
            text = text.replace(
                "#pragma once", '#pragma once\n\n#include "hip_compat.cuh"', 1
            )
    return text


def _rewrite_ridge(text: str) -> str:
    start = text.index("double dpa4_fp32_ridge() {")
    end = text.index("\n}\n", start) + 3
    return text[:start] + _RIDGE + "\n" + text[end:]


def lint(paths) -> list[str]:
    """Return ``path:line token`` for every CUDA-only token left in code.

    Comments are stripped first: the upstream headers mention ``__shfl_sync``
    when explaining the lane layout, and that mention must survive the port.
    """
    offenders = []
    for path in paths:
        path = Path(path)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("//", 1)[0]
            for token in FORBIDDEN:
                if token in code:
                    offenders.append(f"{path.name}:{line_number} {token}")
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deepmd-kit",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "deepmd-kit",
        help="checkout containing source/op/pt/dpa4",
    )
    parser.add_argument("--work", type=Path, default=Path("/tmp/so2_conv_hipify"))
    args = parser.parse_args(argv)

    source = args.deepmd_kit / "source" / "op" / "pt" / "dpa4"
    if not source.is_dir():
        print(f"missing deepmd-kit sources: {source}", file=sys.stderr)
        return 2

    produced = _hipify(source, args.work)
    written = []
    for name, path in sorted(produced.items()):
        # hipify leaves the pre-translation copies next to its output; only the
        # translated files belong in the port tree.
        if not (name.endswith(".hip") or "_hip." in name):
            continue
        text = path.read_text()
        target_name = name
        if target_name == "so2_conv.cu":
            target_name = "so2_conv.hip"
        text = _patch(text, stamp='#include "hip_compat.cuh"')
        if target_name == "so2_conv.hip":
            text = _rewrite_ridge(text)
            text = _restrict_shapes(text)
            text = text.replace(
                '// !!! This is a file automatically generated by hipify!!!',
                '// Translated from deepmd-kit source/op/pt/dpa4/so2_conv.cu by\n'
                '// tools/hipify_so2_conv.py -- regenerate rather than edit.',
            )
        target = OUT / target_name
        target.write_text(text)
        written.append(target.name)

    offenders = lint([OUT / name for name in written])
    for offender in offenders:
        print(f"leftover CUDA token: {offender}", file=sys.stderr)
    print("wrote " + ", ".join(written))
    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
