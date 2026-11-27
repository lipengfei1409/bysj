"""Run the host test suite without having to remember PYTHONPATH.

The Python packages live under `python/` (`kernels`, `dmk`, `reference`,
`model`), so the suite needs that directory on `sys.path`.  Forgetting it is the
one failure that looks like a broken test:

    ModuleNotFoundError: No module named 'dmk'

This runner puts it there and then behaves like `unittest discover`:

    python3 tests/run_python_tests.py                  # everything
    python3 tests/run_python_tests.py -v               # verbose
    python3 tests/run_python_tests.py -k persistent    # only matching tests

It also puts `tests/dcu/` on the path, because a few host tests import helpers
from there (`rocblas_cpp`), and those live next to the DCU harness rather than in
a package.

`--dcu` adds the device suites (`tests/csrc/accuracy`, which skip themselves on a
host), and `--latency` adds the opt-in timing sweep with the rocBLAS baseline::

    python3 tests/run_python_tests.py                 # host only
    python3 tests/run_python_tests.py --dcu           # + accuracy on the device
    python3 tests/run_python_tests.py --dcu --latency # + timing against rocBLAS
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _prepare_path() -> None:
    for extra in (ROOT / "python", ROOT / "tests" / "dcu"):
        path = str(extra)
        if path not in sys.path:
            sys.path.insert(0, path)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _prepare_path()

    verbosity = 2 if ("-v" in argv or "--verbose" in argv) else 1
    pattern = "test_*.py"
    if "-k" in argv:
        index = argv.index("-k")
        try:
            pattern = f"test_*{argv[index + 1]}*.py"
        except IndexError:
            print("-k needs a substring", file=sys.stderr)
            return 2

    into = [ROOT / "tests" / "python"]
    if "--dcu" in argv:
        into += [ROOT / "tests" / "csrc" / "accuracy"]
    if "--latency" in argv:
        into += [ROOT / "tests" / "csrc" / "latency"]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for directory in into:
        try:
            suite.addTests(loader.discover(str(directory), pattern=pattern))
        except ImportError as failure:  # pragma: no cover - broken checkout
            print(f"cannot discover {directory}: {failure}", file=sys.stderr)
            return 2
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
