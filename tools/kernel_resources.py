#!/usr/bin/env python3
"""Report each kernel's VGPR/SGPR/LDS use, and what that does to occupancy.

`hipprof --codeobj-analyze` asks its questions interactively, which is awkward to put
in a runbook.  The same numbers are in the amdgpu metadata the compiler leaves in the
extracted code object, so this reads them straight out of the ELF:

    python3 tools/kernel_resources.py                 # found next to python/kernels/*.so
    python3 tools/kernel_resources.py path/to/x.out   # a specific code object

Occupancy is then arithmetic: a CU here has 64 KiB of LDS, and a wave64 block of
`threads` threads needs `threads * vgpr` VGPRs out of the 65536 a CU has.

The code objects are produced by hipify+hipcc during the build, and `roc-obj-extract`
(or `hipprof --codeobj-analyze`) writes them next to the `.so` as
`*.so-hipv4-amdgcn-amd-amdhsa--gfxNNN.out`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The amdgpu metadata YAML keys worth reporting.
FIELDS = (
    "Name",
    "NumVgprs",
    "NumSgprs",
    "GroupSegmentFixedSize",
    "PrivateSegmentFixedSize",
    "KernargSegmentSize",
    "MaxFlatWorkgroupSize",
)

#: LDS and VGPR budget of one CU on the supported parts.
LDS_BYTES_PER_CU = 64 * 1024
VGPRS_PER_CU = 65536
WAVE = 64


def _candidates(explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(item) for item in explicit]
    found = []
    for pattern in ("python/kernels/*.out", "*.out", "build/**/*.out"):
        found.extend(sorted(ROOT.glob(pattern)))
    return found


def parse(text: str) -> list[dict[str, int | str]]:
    """Pull one record per kernel out of the metadata YAML."""
    kernels: list[dict[str, int | str]] = []
    current: dict[str, int | str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        match = re.match(r"-?\s*([A-Za-z]+):\s*(.+)$", stripped)
        if match is None:
            continue
        key, value = match.group(1), match.group(2).strip()
        if key not in FIELDS:
            continue
        if key == "Name":
            current = {"Name": value}
            kernels.append(current)
            continue
        if current is None:
            continue
        try:
            current[key] = int(value, 0)
        except ValueError:
            current[key] = value
    return kernels


def occupancy(vgprs: int, threads: int, lds_bytes: int = 0) -> int:
    """Blocks per CU: limited by registers, by LDS, and by the CU's wave capacity."""
    if vgprs <= 0 or threads <= 0:
        return 0
    by_vgpr = VGPRS_PER_CU // (threads * vgprs)
    by_lds = LDS_BYTES_PER_CU // lds_bytes if lds_bytes > 0 else 99
    by_waves = 64 // (threads // WAVE)  # 64 waves per CU on gfx9
    return max(0, min(by_vgpr, by_lds, by_waves))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("code_objects", nargs="*",
                        help="extracted *.out code objects (default: search the tree)")
    parser.add_argument("--threads", type=int, default=256,
                        help="block size to evaluate occupancy for (default 256)")
    parser.add_argument("--lds", type=int, default=24576,
                        help="LDS bytes per block of the kernel (default 24 KiB)")
    args = parser.parse_args(argv)

    paths = [path for path in _candidates(args.code_objects) if path.is_file()]
    if not paths:
        print("no extracted code object found; run\n"
              "  python3 -c \"from kernels.native import _extension; _extension('gemm_dense')\"\n"
              "or `hipprof --codeobj-analyze <so>` once, which writes *.out next to it",
              file=sys.stderr)
        return 1

    for path in paths:
        kernels = parse(path.read_text(errors="ignore"))
        if not kernels:
            continue
        print(f"\n{path.name}")
        print(f"  {'kernel':58s} {'vgpr':>5s} {'sgpr':>5s} {'lds':>7s} {'blocks/CU':>9s}")
        for kernel in kernels:
            name = str(kernel.get("Name", "?"))
            short = name if len(name) <= 58 else "…" + name[-57:]
            vgpr = int(kernel.get("NumVgprs", 0) or 0)
            sgpr = int(kernel.get("NumSgprs", 0) or 0)
            lds = int(kernel.get("GroupSegmentFixedSize", 0) or 0)
            total_lds = lds + args.lds
            blocks = occupancy(vgpr, args.threads, total_lds)
            print(f"  {short:58s} {vgpr:5d} {sgpr:5d} {total_lds:7d} {blocks:9d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
