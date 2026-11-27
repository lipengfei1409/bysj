"""Python modeling and wrapper tests.

Same path bootstrap as `tests/__init__.py`, repeated here because
``python3 -m unittest tests.python.test_x`` can load this package on its own
(after ``tests``, which already did it -- but a direct ``import
tests.python`` would otherwise depend on that ordering).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

for _extra in (_ROOT / "python", _ROOT / "tests" / "dcu"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

del _extra, _ROOT
