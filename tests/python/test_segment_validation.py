"""Host contracts for the fused segment validators.

``kernels/segment.py`` validates dtype, shape, device and contiguity *before*
touching the DCU, and those checks are pure Python -- they run on a CPU host.
That matters because the operator tests in ``tests/csrc`` only execute on a DCU:
a validator that crashes on a valid CPU input (for example by iterating over an
optional argument that is ``None``) used to ship unnoticed.
"""
from __future__ import annotations

import unittest

import torch

from kernels.segment import (
    _validate_force_assembly,
    _validate_order,
    _validate_segment_softmax,
    _validate_segment_sum,
    force_assembly,
    segment_softmax,
    segment_sum,
)


def _graph():
    dst = torch.tensor([0, 0, 1, 2, 2])
    offsets = torch.tensor([0, 2, 3, 5])
    return dst, offsets


class SegmentSumValidationTests(unittest.TestCase):
    def test_optional_arguments_may_all_be_none(self):
        dst, offsets = _graph()
        src = torch.randn(5, 4)
        self.assertEqual(_validate_segment_sum(src, dst, offsets, None, None, None), 3)

    def test_optional_arguments_may_all_be_present(self):
        dst, offsets = _graph()
        src = torch.randn(5, 4)
        edge_scale = torch.rand(5)
        node_scale = torch.rand(3)
        order = torch.arange(5)
        self.assertEqual(
            _validate_segment_sum(src, dst, offsets, edge_scale, node_scale, order), 3
        )

    def test_order_alone_is_accepted(self):
        dst, offsets = _graph()
        self.assertEqual(
            _validate_segment_sum(torch.randn(5, 4), dst, offsets, None, None, torch.arange(5)),
            3,
        )

    def test_edge_count_mismatch_is_rejected(self):
        dst, offsets = _graph()
        with self.assertRaises(ValueError):
            _validate_segment_sum(torch.randn(4, 4), dst, offsets, None, None, None)


class SegmentSoftmaxValidationTests(unittest.TestCase):
    def test_optional_arguments_may_all_be_none(self):
        dst, offsets = _graph()
        logit = torch.randn(5, 6)
        env = torch.rand(5)
        self.assertEqual(_validate_segment_softmax(logit, env, offsets, None, None), (3, 6))

    def test_optional_arguments_may_all_be_present(self):
        _, offsets = _graph()
        logit = torch.randn(5, 6)
        env = torch.rand(5)
        null_logit = torch.randn(3, 6)
        order = torch.arange(5)
        self.assertEqual(
            _validate_segment_softmax(logit, env, offsets, null_logit, order), (3, 6)
        )

    def test_null_logit_shape_is_checked(self):
        _, offsets = _graph()
        with self.assertRaises(ValueError):
            _validate_segment_softmax(torch.randn(5, 6), torch.rand(5), offsets, torch.randn(2, 6), None)


class OrderValidationTests(unittest.TestCase):
    def test_wrong_dtype_and_shape_are_rejected(self):
        with self.assertRaises(TypeError):
            _validate_order(torch.arange(5, dtype=torch.float32), 5, "probe")
        with self.assertRaises(ValueError):
            _validate_order(torch.arange(4), 5, "probe")
        self.assertIsNone(_validate_order(None, 5, "probe"))


class ForceAssemblyValidationTests(unittest.TestCase):
    def test_valid_inputs_are_accepted(self):
        dst, dst_offsets = _graph()
        src = torch.tensor([2, 0, 2, 1, 0])
        _, src_offsets = _graph()
        order = torch.tensor([1, 4, 3, 0, 2])
        self.assertEqual(
            _validate_force_assembly(
                torch.randn(5, 3), torch.randn(5, 3), dst_offsets, src_offsets, order
            ),
            3,
        )


class PublicEntryPointTests(unittest.TestCase):
    """Well-formed CPU inputs must reach the DCU guard, not fail in validation."""

    def test_segment_sum_stops_at_the_dcu_guard(self):
        dst, offsets = _graph()
        with self.assertRaisesRegex(RuntimeError, "DCU"):
            segment_sum(torch.randn(5, 4), dst, offsets, None, None)

    def test_segment_softmax_stops_at_the_dcu_guard(self):
        _, offsets = _graph()
        with self.assertRaisesRegex(RuntimeError, "DCU"):
            segment_softmax(torch.randn(5, 6), torch.rand(5), offsets)

    def test_force_assembly_stops_at_the_dcu_guard(self):
        _, dst_offsets = _graph()
        src_offsets = torch.tensor([0, 2, 3, 5])
        with self.assertRaisesRegex(RuntimeError, "DCU"):
            force_assembly(
                torch.randn(5, 3),
                torch.randn(5, 3),
                torch.tensor([2, 0, 2, 1, 0]),
                torch.tensor([0, 0, 1, 2, 2]),
                dst_offsets,
                src_offsets,
                torch.tensor([1, 4, 3, 0, 2]),
            )


if __name__ == "__main__":
    unittest.main()
