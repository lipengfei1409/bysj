"""Tests for the standalone TileLang DPA4 edge-type feature operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops import (
    edge_type_features_reference,
    tilelang_edge_type_features,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class EdgeTypeFeatureTests(unittest.TestCase):
    def test_reference_gathers_sums_and_masks(self):
        node_features = torch.arange(15, dtype=torch.float64).reshape(5, 3)
        src = torch.tensor([0, 1, 4, 2], dtype=torch.int64)
        dst = torch.tensor([2, 1, 3, 0], dtype=torch.int64)
        keep = torch.tensor([[1.0], [0.0], [1.0], [0.25]], dtype=torch.float64)
        actual = edge_type_features_reference(node_features, src, dst, keep)
        expected = (node_features[src] + node_features[dst]) * keep
        torch.testing.assert_close(actual, expected)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        features = torch.zeros((3, 4), dtype=torch.float32)
        indices = torch.zeros((2,), dtype=torch.int64)
        keep = torch.ones((2, 1), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_edge_type_features(features, indices, indices, keep)

    def test_invalid_inputs_are_rejected(self):
        features = torch.zeros((3, 4), dtype=torch.float32)
        indices = torch.zeros((2,), dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, r"shape \(n_edges, 1\)"):
            tilelang_edge_type_features(
                features, indices, indices, torch.ones((2,), dtype=torch.float32)
            )
        with self.assertRaisesRegex(TypeError, "torch.int64"):
            tilelang_edge_type_features(
                features,
                indices.to(torch.int32),
                indices,
                torch.ones((2, 1), dtype=torch.float32),
            )

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_matches_reference_without_parameter_gradient(self):
        features = torch.randn((11, 7), dtype=torch.float32, device="cuda")
        features.requires_grad_()
        src = torch.tensor([0, 3, 4, 7, 2], dtype=torch.int64, device="cuda")
        dst = torch.tensor([1, 3, 5, 8, 0], dtype=torch.int64, device="cuda")
        keep = torch.tensor(
            [[1.0], [0.0], [0.4], [1.0], [0.8]], dtype=torch.float32, device="cuda"
        )
        actual = tilelang_edge_type_features(features, src, dst, keep)
        expected = edge_type_features_reference(features, src, dst, keep)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(
            torch.autograd.grad(actual.sum(), features, allow_unused=True)[0]
        )


if __name__ == "__main__":
    unittest.main()
