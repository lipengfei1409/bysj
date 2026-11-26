"""Structural tests for the ops/ + dmk/ split.

``csrc/ops/`` owns the operator templates and the factory; ``csrc/dmk/`` owns
what persistent scheduling needs.  The split is only worth anything if it holds:
an op must not reach into the runtime, the runtime must not own op templates, and
every pair that shares arithmetic must share it through a header rather than by
duplicating it.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "csrc" / "ops"
DMK = ROOT / "csrc" / "dmk"


def _headers(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.h"))


class LayoutTests(unittest.TestCase):
    def test_ops_holds_the_templates_and_the_factory(self) -> None:
        names = {p.name for p in _headers(OPS)}
        for expected in ("op.h", "config.h", "demo_op.h", "gemm.h", "tile.h",
                         "lds.h", "so2_conv_op.h", "so2_conv_math.h"):
            self.assertIn(expected, names)

    def test_dmk_holds_only_the_persistent_runtime(self) -> None:
        names = {p.name for p in _headers(DMK)}
        self.assertEqual(names, {"megakernel.h", "state.h", "controller.h"})
        self.assertTrue((DMK / "megakernel.cpp").is_file())

    def test_ops_depend_on_the_state_but_not_on_the_scheduler(self) -> None:
        # A hook's parameter is `State&`, so ops legitimately include
        # dmk/state.h; what they must never reach for is the driver.
        for header in _headers(OPS):
            text = header.read_text()
            for forbidden in ("dmk/megakernel.h", "dmk/controller.h"):
                self.assertNotIn(f'#include "{forbidden}"', text, header.name)

    def test_the_runtime_may_include_ops(self) -> None:
        # The VM drives ops, so the dependency points this way only.
        text = (DMK / "megakernel.h").read_text()
        self.assertIn('#include "ops/', text)

    def test_includes_name_their_folder(self) -> None:
        # A bare `#include "op.h"` would resolve only by luck once the files are
        # in two directories; every cross-folder include is spelled out.
        known = {p.name for p in _headers(OPS)} | {p.name for p in _headers(DMK)}
        for directory in (OPS, DMK):
            for path in list(directory.glob("*.h")) + list(directory.glob("*.cpp")):
                for match in re.finditer(r'#include "([A-Za-z0-9_./-]+)"', path.read_text()):
                    target = match.group(1)
                    if "/" not in target:
                        self.assertNotIn(target, known, f"{path.name}: bare {target}")
                    else:
                        self.assertTrue((OPS.parent / target).is_file(), f"{path.name}: {target}")


ORDINARY = ROOT / "csrc" / "so2_conv.cpp"


class TwoEntryPointTests(unittest.TestCase):
    def test_both_entries_share_one_math_header(self) -> None:
        op = (OPS / "so2_conv_op.h").read_text()
        kernel = ORDINARY.read_text()
        for text, name in ((op, "so2_conv_op.h"), (kernel, "so2_conv.cpp")):
            self.assertIn('#include "ops/so2_conv_math.h"', text, name)

    def test_neither_entry_reimplements_the_span(self) -> None:
        # The gate expression exists exactly once, in the shared header.
        math = (OPS / "so2_conv_math.h").read_text()
        self.assertIn("sigmoid_content", math)
        for text, name in (
            ((OPS / "so2_conv_op.h").read_text(), "so2_conv_op.h"),
            (ORDINARY.read_text(), "so2_conv.cpp"),
        ):
            self.assertNotIn("__expf", text, name)

    def test_only_the_ordinary_entry_point_is_wired(self) -> None:
        # Decision for now: reachable through the normal operator path, and the
        # persistent op stays source-only until the runtime is wired.
        self.assertTrue(ORDINARY.is_file())
        top = (ROOT / "setup.py").read_text()
        self.assertIn('"csrc/so2_conv.cpp"', top)
        self.assertIn('include_dirs=["csrc"]', top)

        self.assertIn("so2_conv_hip", (ROOT / "csrc" / "ops.h").read_text())
        self.assertIn('m.def("so2_conv"', (ROOT / "csrc" / "bindings.cpp").read_text())
        self.assertIn("so2_conv", (ROOT / "python" / "kernels" / "__init__.py").read_text())
        self.assertTrue((ROOT / "python" / "kernels" / "so2_conv.py").is_file())

    def test_the_persistent_op_is_not_built(self) -> None:
        # So2ConvOp has no launch path yet, so nothing should compile it and no
        # dmk-side build should exist.
        for path in (ROOT / "setup.py", ROOT / "csrc" / "ops.h",
                     ROOT / "csrc" / "bindings.cpp"):
            self.assertNotIn("So2ConvOp", path.read_text(), path.name)
        self.assertFalse((DMK / "setup.py").exists())
        self.assertFalse((ROOT / "python" / "kernels" / "dmk").exists())


if __name__ == "__main__":
    unittest.main()
