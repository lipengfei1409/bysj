"""Compatibility import for the canonical model.dpa4.neighbor_bridge module."""
from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("model.dpa4.neighbor_bridge")
sys.modules[__name__] = _module
