"""Run the test suites without having to remember PYTHONPATH.

The Python packages live under `python/` (`kernels`, `dmk`, `reference`, `model`),
so the suite needs that directory on `sys.path`.  Forgetting it is the one failure
that looks like a broken test:

    ModuleNotFoundError: No module named 'dmk'

This runner puts it there and then behaves like `unittest discover`:

    python3 tests/run_python_tests.py                  # host suite
    python3 tests/run_python_tests.py -v               # verbose
    python3 tests/run_python_tests.py -k persistent    # only matching tests

The device suites live under `tests/csrc/` and are discovered by name rather than by
package (they are plain modules):

    python3 tests/run_python_tests.py --dcu            # + accuracy (skips on a host)
    python3 tests/run_python_tests.py --dcu --latency  # + timing against rocBLAS
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Suites, by flag.  `host` is always run.
SUITES = {
    "host": ("tests/python", "test_*.py"),
    "--dcu": ("tests/csrc/accuracy", "test_*.py"),
    "--latency": ("tests/csrc/latency", "test_*.py"),
}


def _prepare_path() -> None:
    for extra in (ROOT / "python", ROOT / "tests" / "dcu", ROOT / "tests" / "csrc"):
        path = str(extra)
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_module(path: Path, name: str):
    """Import a test module from its path.

    `unittest discover` refuses a start directory outside the project root, and
    `tests/csrc/accuracy` is not a package, so the modules are loaded directly.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _suite_for(relative: str, pattern: str) -> unittest.TestSuite:
    directory = ROOT / relative
    suite = unittest.TestSuite()
    for path in sorted(directory.glob(pattern)):
        module = _load_module(path, f"_dpa4_{path.stem}")
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    return suite


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _prepare_path()

    verbosity = 2 if ("-v" in argv or "--verbose" in argv) else 1
    substring = None
    if "-k" in argv:
        index = argv.index("-k")
        try:
            substring = argv[index + 1]
        except IndexError:
            print("-k needs a substring", file=sys.stderr)
            return 2

    selected = ["host"] + [flag for flag in SUITES if flag != "host" and flag in argv]
    suite = unittest.TestSuite()
    for name in selected:
        relative, pattern = SUITES[name]
        try:
            part = _suite_for(relative, pattern)
        except ImportError as failure:  # pragma: no cover - broken checkout
            print(f"cannot discover {relative}: {failure}", file=sys.stderr)
            return 2
        if substring is not None:
            part = _filter(part, substring)
        suite.addTests(part)

    print("suites: " + ", ".join(selected))
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


def _filter(suite: unittest.TestSuite, substring: str) -> unittest.TestSuite:
    """Keep the tests whose id contains `substring` (keeps sub-suites)."""
    kept = unittest.TestSuite()
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            inner = _filter(test, substring)
            if inner.countTestCases():
                kept.addTest(inner)
        elif substring in test.id():
            kept.addTest(test)
    return kept


if __name__ == "__main__":
    sys.exit(main())
