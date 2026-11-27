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
        for expected in ("op.h", "config.h", "elementwise.h", "gemm.h", "tile.h",
                         "lds.h", "so2_conv_op.h", "so2_conv_math.h"):
            self.assertIn(expected, names)

    def test_dmk_holds_only_the_persistent_runtime(self) -> None:
        names = {p.name for p in _headers(DMK)}
        self.assertEqual(names, {"megakernel.h", "state.h", "controller.h"})
        self.assertTrue((DMK / "megakernel.cpp").is_file())

    def test_ops_never_include_anything_under_dmk(self) -> None:
        # The rule, no exceptions: an operator header may not depend on the
        # runtime.  The shared-memory contract it *is* written against lives in
        # ops/state.h, so `ops` stays self-contained.
        for header in _headers(OPS):
            text = header.read_text()
            self.assertNotIn('#include "dmk/', text, header.name)

    def test_the_runtime_may_include_ops(self) -> None:
        # The VM drives ops, so the dependency points this way only.
        text = (DMK / "megakernel.h").read_text()
        self.assertIn('#include "ops/', text)

    def test_ops_live_in_the_ops_namespace(self) -> None:
        # Everything under csrc/ops/ is declared in ops:: -- the operator
        # templates, the config contract, the LDS and tensor helpers.  The one
        # exception is the forward declaration of the runtime's per-wave State,
        # which has to name its real home (dmk::mk).
        for header in _headers(OPS):
            text = header.read_text()
            # No dmk:: namespace may be opened here, not even for the State
            # forward declaration: that lives in ops/state.h now.
            self.assertEqual(
                re.findall(r"\nnamespace (dmk::[A-Za-z_:]+) \{", text), [], header.name
            )

    def test_the_state_contract_lives_with_the_ops(self) -> None:
        # `State` is the interface between runtime and operator, so it is
        # declared in ops/state.h and hooks take a bare `State&`.
        template = (OPS / "op.h").read_text()
        self.assertIn("struct State;", template)
        self.assertNotIn("dmk::", template)
        self.assertIn("State& state", template)
        contract = (OPS / "state.h").read_text()
        self.assertIn("struct State {", contract)
        for header in _headers(OPS):
            self.assertNotIn("dmk::", header.read_text(), header.name)

    def test_the_runtime_aliases_the_ops_namespace(self) -> None:
        for path in (DMK / "state.h", DMK / "megakernel.h", DMK / "controller.h"):
            text = path.read_text()
            self.assertIn("namespace config = ops::config;", text, path.name)
            self.assertIn('namespace dmk::mk {', text, path.name)

    def test_the_runtime_builds_on_the_ops_contract(self) -> None:
        text = (DMK / "state.h").read_text()
        self.assertIn('#include "ops/state.h"', text)
        self.assertIn("using ops::State;", text)

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


class ModeSwitchTests(unittest.TestCase):
    def test_the_switch_is_a_single_compile_time_define(self) -> None:
        mode = (OPS / "mode.h").read_text()
        self.assertIn("#define OPS_MODE_PERSISTENT 0", mode)
        self.assertIn("#define OPS_MODE_ORDINARY 1", mode)
        self.assertIn("#ifndef OPS_MODE", mode)
        self.assertIn("#define OPS_MODE OPS_MODE_PERSISTENT", mode)  # default
        self.assertIn('"OPS_MODE must be OPS_MODE_PERSISTENT (0) or OPS_MODE_ORDINARY (1)"', mode)
        # The two forms must never be compiled into one translation unit.
        self.assertIn("constexpr bool kPersistentMode", mode)

    def test_the_converted_operator_has_both_forms_in_one_header(self) -> None:
        op = (OPS / "so2_conv_op.h").read_text()
        self.assertIn("#if OPS_MODE == OPS_MODE_PERSISTENT", op)
        self.assertIn("#if OPS_MODE == OPS_MODE_ORDINARY", op)
        self.assertIn("class Op final : public BaseOp<Globals>", op)
        self.assertIn("__global__ void so2_conv_span_kernel", op)
        # Both forms on one class.
        self.assertIn("struct So2Conv {", op)
        self.assertIn("static void launch(", op)

    def test_the_ordinary_mode_does_not_see_the_runtime_contract(self) -> None:
        # Ordinary mode must compile without ops/state.h or ops/op.h: no pages,
        # no hooks, no State.
        op = (OPS / "so2_conv_op.h").read_text()
        head = op[: op.index("namespace ops::so2_conv")]
        self.assertIn("#if OPS_MODE == OPS_MODE_PERSISTENT", head)
        self.assertIn('#include "ops/state.h"', head)
        ordinary = head[head.index("#else") :]
        self.assertIn("#include <hip/hip_runtime.h>", ordinary)
        self.assertNotIn("ops/state.h", ordinary)

    def test_the_ordinary_translation_unit_only_selects_a_mode(self) -> None:
        kernel = (ROOT / "csrc" / "so2_conv.cpp").read_text()
        self.assertIn("#define OPS_MODE OPS_MODE_ORDINARY", kernel)
        self.assertIn('#include "ops/so2_conv_op.h"', kernel)
        # The kernel body moved into the header, so the launcher has no __global__.
        self.assertNotIn("__global__", kernel)


class TwoEntryPointTests(unittest.TestCase):
    def test_both_modes_share_one_math_header(self) -> None:
        # Both forms live in one header now, so the invariant is checked once:
        # the header pulls in the shared arithmetic and the torch glue only
        # selects a mode.
        op = (OPS / "so2_conv_op.h").read_text()
        self.assertIn('#include "ops/so2_conv_math.h"', op)
        self.assertIn("#define OPS_MODE OPS_MODE_ORDINARY", ORDINARY.read_text())

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
