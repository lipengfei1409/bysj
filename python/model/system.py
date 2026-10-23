"""Compatibility import for the canonical model.dpa4.system module."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("model.dpa4.system")
sys.modules[__name__] = _module
