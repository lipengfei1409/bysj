"""Contract tests for the megakernel SO(2) conv operator.

The op is device code, so this checks the parts a host can check: that the class
follows the dmk op template (`BaseOp`'s five hooks, a static opcode, a device
constructor taking the instruction words), and that the page budget and
instruction encoding are consistent with what the header documents.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OP = ROOT / "csrc" / "so2_conv_dmk" / "so2_conv_op.h"
TEMPLATE = ROOT / "csrc" / "dmk" / "op.h"


class TemplateConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = OP.read_text()

    def test_it_derives_from_the_base_op_with_a_static_opcode(self) -> None:
        self.assertIn("class So2ConvOp final : public BaseOp<Globals>", self.text)
        self.assertIn("static constexpr int opcode", self.text)
        self.assertIn("inline constexpr int kSo2ConvOpcode", self.text)

    def test_all_five_template_hooks_are_overridden(self) -> None:
        for hook in ("release_lid", "init_semaphores", "load", "run", "store"):
            self.assertRegex(self.text, rf"__device__ .*\b{hook}\(", hook)
            self.assertIn(f"override", self.text)
        # Every hook the base declares must be implemented here.
        base = TEMPLATE.read_text()
        declared = set(re.findall(r"virtual \w+ (\w+)\(", base))
        self.assertEqual(
            declared, {"release_lid", "init_semaphores", "load", "run", "store"}
        )
        for hook in declared:
            self.assertIn(hook, self.text, hook)

    def test_it_is_trivially_destructible_like_the_template_requires(self) -> None:
        # No user-declared destructor, and no owning member.
        self.assertNotIn("~So2ConvOp", self.text)

    def test_the_constructor_takes_the_instruction_words(self) -> None:
        self.assertIn("explicit So2ConvOp(const int* instruction)", self.text)

    def test_opcode_does_not_collide_with_the_reference_op(self) -> None:
        demo = (ROOT / "csrc" / "dmk" / "demo_op.h").read_text()
        demo_opcode = int(re.search(r"opcode = (\d+)", demo).group(1))
        ours = int(re.search(r"kSo2ConvOpcode = (\d+)", self.text).group(1))
        self.assertNotEqual(demo_opcode, ours)


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = OP.read_text()

    def test_the_page_budget_is_pinned_for_the_profiled_shape(self) -> None:
        # 2*red*dim + 2*layers*red*red + 1 floats, at (dim=16, red=10, layers=4).
        floats = 2 * 10 * 16 + 2 * 4 * 10 * 10 + 1
        self.assertEqual(floats, 1121)
        self.assertLessEqual(floats * 4, 8192)
        self.assertIn("so2_conv_table_bytes(16, 10, 4) <= config::kPageSize", self.text)

    def test_the_instruction_layout_is_documented_and_encoded(self) -> None:
        # The instruction block comes first; the semaphore block below it also
        # uses bracketed indices, so only the leading seven are the instruction.
        words = re.findall(r"^//   \[(\d)\] (\w+)", self.text, re.M)[:7]
        self.assertEqual([int(i) for i, _ in words], [0, 1, 2, 3, 4, 5, 6])
        for index, name in words:
            self.assertIn(f"words[{index}]", self.text, name)

    def test_semaphores_match_the_documented_count(self) -> None:
        self.assertIn("s.semaphores()[0].init(1);", self.text)
        self.assertIn("s.semaphores()[1].init(config::kConsumerWaves);", self.text)
        self.assertIn("return 2;", self.text)

    def test_the_lane_axis_is_the_channel_axis(self) -> None:
        # The whole reason for the 64-lane rewrite: lane t is channel t.
        self.assertIn("kLaneChannels = config::kWaveSize", self.text)
        self.assertIn("const int channel = lane;", self.text)

    def test_the_unimplemented_parts_are_named(self) -> None:
        self.assertIn("Not in this op, on purpose:", self.text)
        for missing in ("destination reduction", "backward", "Wigner run packing"):
            self.assertIn(missing, self.text, missing)


if __name__ == "__main__":
    unittest.main()
