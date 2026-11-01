"""Tests for the standalone TileLang DPA4 edge geometry operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops import edge_geometry_reference, tilelang_edge_geometry

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class EdgeGeometryTests(unittest.TestCase):
    def test_reference_matches_regularized_norm_and_gradients(self):
        edge_vec = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        length, unit = edge_geometry_reference(edge_vec, eps=1.0e-7)
        expected_length = torch.sqrt(
            (edge_vec * edge_vec).sum(-1, keepdim=True) + 1e-14
        )
        torch.testing.assert_close(length, expected_length)
        torch.testing.assert_close(unit, edge_vec / expected_length)
        loss = length.square().sum() + unit.square().sum()
        (gradient,) = torch.autograd.grad(loss, edge_vec)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_edge_geometry(torch.zeros((2, 3), dtype=torch.float32))

    def test_invalid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"shape \(n_edges, 3\)"):
            tilelang_edge_geometry(torch.zeros((2, 4), dtype=torch.float32))

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        edge_vec = torch.randn((37, 3), dtype=torch.float32, device="cuda")
        edge_vec[0] = 0.0
        edge_vec.requires_grad_()
        grad_len = torch.randn((37, 1), dtype=torch.float32, device="cuda")
        grad_unit = torch.randn_like(edge_vec)

        actual = tilelang_edge_geometry(edge_vec)
        actual_grad = torch.autograd.grad(actual, edge_vec, (grad_len, grad_unit))[0]
        reference_vec = edge_vec.detach().clone().requires_grad_(True)
        expected = edge_geometry_reference(reference_vec)
        expected_grad = torch.autograd.grad(
            expected, reference_vec, (grad_len, grad_unit)
        )[0]
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=2.0e-5, atol=2.0e-6
            )
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3.0e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
