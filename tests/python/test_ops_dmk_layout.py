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
                         "lds.h", "state.h"):
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


class SpecificOptimisationTests(unittest.TestCase):
    """`csrc/ops/` is for generic persistent ops; specific work is in `csrc/`."""

    def test_the_specific_optimisations_live_in_csrc(self) -> None:
        self.assertTrue((ROOT / "csrc" / "so2_conv.cpp").is_file())
        self.assertTrue((ROOT / "csrc" / "so2_conv_math.h").is_file())
        self.assertTrue((ROOT / "csrc" / "gie.cpp").is_file())  # GIE / zonal scatter
        for gone in ("so2_conv_op.h", "so2_conv_math.h", "mode.h"):
            self.assertFalse((OPS / gone).exists(), gone)

    def test_ops_holds_only_the_generic_operators(self) -> None:
        ops = {p.name for p in _headers(OPS)}
        self.assertEqual(
            ops,
            {
                "op.h", "state.h", "config.h", "lds.h", "copy.h", "tensor.h",
                "tile.h", "dtype.h", "dumma.h", "utils.h",
                "noop.h", "elementwise.h", "embedding.h", "gemm.h", "attention.h",
            },
        )

    def test_gemm_has_exactly_one_class(self) -> None:
        # The DUMMA-fragment GEMM used to live beside GemmOp and was never called
        # by it; two implementations of one operator is one too many.
        gemm = (OPS / "gemm.h").read_text()
        self.assertIn("class GemmOp final : public BaseOp<Globals>", gemm)
        for gone in ("GemmWaveT", "GemmSharedT", "GemmParamsT", "run_gemm_wave",
                     "mma_supported", "dumma"):
            self.assertNotIn(gone, gemm, gone)
        self.assertEqual(gemm.count("class "), 1)

    def test_no_operator_carries_a_mode_switch_any_more(self) -> None:
        for header in _headers(OPS):
            text = header.read_text()
            self.assertNotIn("OPS_MODE", text, header.name)

    def test_the_conv_kernel_is_a_plain_launch(self) -> None:
        kernel = (ROOT / "csrc" / "so2_conv.cpp").read_text()
        self.assertIn("__global__ void so2_conv_kernel", kernel)
        self.assertIn("<<<blocks_for(edges), kBlockThreads", kernel)
        self.assertIn('#include "so2_conv_math.h"', kernel)
        # No pages, no semaphores, no BaseOp: it is not a persistent op.
        for forbidden in ("State&", "semaphores()", "BaseOp"):
            self.assertNotIn(forbidden, kernel)

    def test_the_conv_arithmetic_exists_once(self) -> None:
        kernel = (ROOT / "csrc" / "so2_conv.cpp").read_text()
        math = (ROOT / "csrc" / "so2_conv_math.h").read_text()
        self.assertIn("sigmoid_content", math)
        self.assertNotIn("__expf", kernel)  # the gate expression lives in the math


