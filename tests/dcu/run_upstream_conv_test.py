#!/usr/bin/env python3
"""Run DeepMD's own ``dpa4_so2_conv`` test suite against this HIP port.

Upstream ships the reference tests for the operator it wrote
(``source/tests/pt/model/test_descriptor_sezm_cuda.py``): forward against a dense
transcription of the math on every zoo shape, backward against autograd,
reduction reproducibility, ``make_fx`` tracing, zero-edge / zero-envelope edges,
and the multi-head rejection.  Those tests gate on
``op_available() == torch.ops.deepmd.dpa4_so2_conv exists``, so loading this
port's extension first is all it takes for them to exercise the HIP translation.

Usage (on a DCU host, after ``source /opt/dtk/env.sh``)::

    DEEPMD_SRC=~/private_data/lipengfei/deepmd-kit \\
        python3 tests/dcu/run_upstream_conv_test.py

    # only the convolution cases, verbose
    python3 tests/dcu/run_upstream_conv_test.py --extra -v --extra -k --extra ConvCuda

The runner prints the location it resolved and the command it runs, so a
missing checkout or a wrong filter is visible before the build starts.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PORT = ROOT / "csrc" / "so2_conv_hip"
UPSTREAM_TEST = Path("source/tests/pt/model/test_descriptor_sezm_cuda.py")
BUILD = PORT / "build.py"


def find_test_file(deepmd_src: str | None = None, *, search_defaults: bool = True) -> Path | None:
    """Resolve the upstream test file from a checkout or the installed package.

    ``search_defaults=False`` restricts the search to the explicitly given
    paths, which is what makes the resolution testable on a machine that also
    happens to have a sibling checkout.
    """
    candidates: list[Path] = []
    if deepmd_src:
        candidates.append(Path(deepmd_src).expanduser())
    env_src = os.environ.get("DEEPMD_SRC")
    if env_src:
        candidates.append(Path(env_src).expanduser())
    # A source checkout next to this repo is the common layout on the cluster.
    if not search_defaults:
        for candidate in candidates:
            test = candidate / UPSTREAM_TEST
            if test.is_file():
                return test
        return None
    candidates.extend(
        [
            ROOT.parent / "deepmd-kit",
            ROOT / "deepmd-kit",
        ]
    )
    for candidate in candidates:
        test = candidate / UPSTREAM_TEST
        if test.is_file():
            return test
    # Installed package: the tests are usually not shipped, but the package
    # location points at the checkout that built it when it is pip -e.
    try:
        import deepmd

        package = Path(deepmd.__file__).resolve().parent
        for ancestor in package.parents:
            test = ancestor / UPSTREAM_TEST
            if test.is_file():
                return test
    except ImportError:
        pass
    return None


def build_argv(test_file: Path, *, extra: list[str] | None = None) -> list[str]:
    """Return the pytest command that loads the port before collecting tests."""
    return [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-q",
        "-k",
        "ConvCuda",
        *(extra or []),
    ]


def child_env() -> dict:
    """Environment for the child: make the plugin importable and preload it."""
    env = dict(os.environ)
    parts = [str(HERE), str(PORT)]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deepmd-src", default=None, help="DeepMD checkout root")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--extra", action="append", default=[], help="extra argument for pytest"
    )
    args = parser.parse_args(argv)

    test_file = find_test_file(args.deepmd_src)
    if test_file is None:
        print(
            "cannot find source/tests/pt/model/test_descriptor_sezm_cuda.py; "
            "pass --deepmd-src or set DEEPMD_SRC",
            file=sys.stderr,
        )
        return 2
    if importlib.util.find_spec("pytest") is None:
        print("these upstream tests are unittest classes but run under pytest; "
              "install pytest in this environment", file=sys.stderr)
        return 2

    if not args.skip_build:
        print(f"building {BUILD}", flush=True)
        code = subprocess.call([sys.executable, str(BUILD)])
        if code != 0:
            print(f"build failed (exit {code})", file=sys.stderr)
            return code

    command = [
        *build_argv(test_file, extra=args.extra),
        "-p",
        "so2_conv_hip_plugin",
    ]
    print(f"test file: {test_file}", flush=True)
    print("$ " + " ".join(command), flush=True)
    return subprocess.call(command, env=child_env())


if __name__ == "__main__":
    sys.exit(main())
