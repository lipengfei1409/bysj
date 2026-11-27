#!/usr/bin/env python3
"""Remove build residue that a plain `setup.py clean` does not know about.

Three things outlive `build/`, and each has cost a wasted build here:

    build/                     ninja objects; stale after an arch change
    csrc/**/*_hip.cpp|.h       torch's hipify writes these *next to the sources*,
                               and a stale copy is compiled instead of the real
                               headers -- nothing in the tree looks wrong
    python/kernels/_C*.so      `build_ext --inplace` overwrites it, `clean` leaves it

This lives in a script rather than as a `setup.py` command on purpose: subclassing
`distutils.command.clean` inside `setup.py` pulls setuptools' vendored distutils in
at import time and, on at least one install here, breaks `setup()` itself with
`MinimalDistribution has no attribute set_defaults`.  A build tool that cannot start
is worse than one extra command.

    python3 tools/clean_build.py            # remove the residue
    python3 tools/clean_build.py --dry-run  # list it
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories removed whole, relative to the repository root.
DIRECTORIES = ("build",)

#: Files removed by glob, relative to the repository root.
FILES = (
    "csrc/**/*_hip.cpp",
    "csrc/**/*_hip.h",
    "python/kernels/_C*.so",
    "python/kernels/_C*.pyd",
)


def collect() -> list[Path]:
    found: list[Path] = []
    for name in DIRECTORIES:
        path = ROOT / name
        if path.is_dir():
            found.append(path)
    for pattern in FILES:
        found.extend(sorted(ROOT.glob(pattern)))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be removed")
    args = parser.parse_args(argv)

    targets = collect()
    if not targets:
        print("nothing to clean")
        return 0
    for path in targets:
        print(("would remove " if args.dry_run else "removing ") + str(path.relative_to(ROOT)))
        if args.dry_run:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print(f"{len(targets)} item(s)" + (" (dry run)" if args.dry_run else " removed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
