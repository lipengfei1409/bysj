"""Host tests for the slot-axis operators: oracles and the validation contract.

The kernels themselves run on a DCU (``tests/csrc/accuracy/test_slot_ops_accuracy.py``);
what can be checked anywhere is that the reference implements the documented
contract and that the wrappers reject malformed inputs before touching the
device.
"""
from __future__ import annotations

import unittest

import torch

from kernels.slot import slot_pack_scale, slot_reduce
from reference.slot import slot_pack_scale_reference, slot_reduce_reference


class SlotPackScaleReferenceTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        src = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        slot = torch.tensor([1, 0, 2, -1, -1, 1], dtype=torch.int64)
        scale = torch.linspace(0.5, 1.5, 2 * 6 * 4).reshape(2, 6, 4)
        return src, slot, scale

    def test_gathers_by_slot_and_zeroes_negative_rows(self):
        src, slot, _ = self._inputs()
        out = slot_pack_scale_reference(src, slot)
        self.assertEqual(tuple(out.shape), (2, 6, 4))
        torch.testing.assert_close(out[:, 0], src[:, 1])
        torch.testing.assert_close(out[:, 1], src[:, 0])
        torch.testing.assert_close(out[:, 2], src[:, 2])
        self.assertEqual(out[:, 3].count_nonzero().item(), 0)
        self.assertEqual(out[:, 4].count_nonzero().item(), 0)
        torch.testing.assert_close(out[:, 5], src[:, 1])

    def test_scale_multiplies_elementwise_after_the_gather(self):
        src, slot, scale = self._inputs()
        out = slot_pack_scale_reference(src, slot, scale)
        gathered = src.index_select(1, slot.clamp_min(0))
        expected = torch.where((slot < 0).reshape(1, 6, 1), torch.zeros_like(gathered), gathered)
        torch.testing.assert_close(out, expected * scale)

    def test_rejects_a_slot_entry_outside_the_source_rows(self):
        src = torch.zeros(2, 3, 4)
        with self.assertRaises(ValueError):
            slot_pack_scale_reference(src, torch.tensor([3], dtype=torch.int64))


class SlotReduceReferenceTests(unittest.TestCase):
    @staticmethod
    def _inputs():
        generator = torch.Generator().manual_seed(11)
        src = torch.randn(5, 6, 4, generator=generator)
        weight = torch.rand(5, 6, generator=generator) + 0.1
        return src, weight

    def test_plain_sum_matches_torch(self):
        src, _ = self._inputs()
        torch.testing.assert_close(slot_reduce_reference(src), src.sum(dim=1))

    def test_weighted_sum_matches_the_explicit_product(self):
        src, weight = self._inputs()
        expected = (src * weight.unsqueeze(-1)).sum(dim=1)
        torch.testing.assert_close(slot_reduce_reference(src, weight), expected)

    def test_single_row_is_the_identity(self):
        src = torch.randn(3, 1, 5)
        torch.testing.assert_close(slot_reduce_reference(src), src[:, 0, :])

    def test_rejects_a_mismatched_weight(self):
        src, _ = self._inputs()
        with self.assertRaises(ValueError):
            slot_reduce_reference(src, torch.ones(5, 5))


class SlotWrapperContractTests(unittest.TestCase):
    """Malformed inputs must fail in the wrapper, not on the device."""

    def test_pack_rejects_bad_slot_dtype_and_range(self):
        src = torch.zeros(2, 3, 4)
        with self.assertRaises(TypeError):
            slot_pack_scale(src, torch.tensor([0.0, 1.0]))
        with self.assertRaises(ValueError):
            slot_pack_scale(src, torch.tensor([0, 3], dtype=torch.int64))
        with self.assertRaises(ValueError):
            slot_pack_scale(src, torch.tensor([[0, 1]], dtype=torch.int64))

    def test_pack_rejects_a_mismatched_scale(self):
        src = torch.zeros(2, 3, 4)
        slot = torch.tensor([0, 1], dtype=torch.int64)
        with self.assertRaises(ValueError):
            slot_pack_scale(src, slot, torch.zeros(2, 3, 4))
        with self.assertRaises(TypeError):
            slot_pack_scale(src, slot, torch.zeros(2, 2, 4).double())

    def test_reduce_rejects_bad_weight_shape(self):
        src = torch.zeros(2, 6, 4)
        with self.assertRaises(ValueError):
            slot_reduce(src, torch.zeros(2, 5))
        with self.assertRaises(TypeError):
            slot_reduce(src, torch.zeros(2, 6).double())

    def test_rejects_non_fp32_src(self):
        with self.assertRaises(TypeError):
            slot_reduce(torch.zeros(2, 6, 4).double())
        with self.assertRaises(TypeError):
            slot_pack_scale(torch.zeros(2, 3, 4).double(), torch.tensor([0], dtype=torch.int64))

    def test_well_formed_cpu_inputs_reach_the_dcu_guard(self):
        src = torch.zeros(2, 3, 4)
        slot = torch.tensor([0, 1], dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "DCU"):
            slot_pack_scale(src, slot)
        with self.assertRaisesRegex(RuntimeError, "DCU"):
            slot_reduce(torch.zeros(2, 6, 4))


if __name__ == "__main__":
    unittest.main()
