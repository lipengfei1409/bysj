"""Project tests.

Importing this package puts the project's Python packages on ``sys.path``.

They live under ``python/`` (``dmk``, ``kernels``, ``reference``, ``model``), which
is not a directory Python searches by default, so every test module would fail
with ``ModuleNotFoundError: No module named 'dmk'`` -- a traceback that reads like
a broken test rather than a missing path.  Doing it here means the naive
invocation works::

    python3 -m unittest tests.python.test_persistent_gemm
    python3 -m unittest discover -s tests -p 'test_*.py'

``tests/run_python_tests.py`` does the same for ``discover -s tests/python``,
where unittest imports the modules as top-level and this package is never loaded.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

for _extra in (_ROOT / "python", _ROOT / "tests" / "dcu"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

del _extra, _ROOT
