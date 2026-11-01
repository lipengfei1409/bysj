"""Tests for the standalone TileLang DPA4 edge quaternion operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops import edge_quaternion_reference, tilelang_edge_quaternion

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class EdgeQuaternionTests(unittest.TestCase):
    def test_reference_handles_chart_poles_and_has_finite_gradients(self):
        edge_vec = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0e-4, -2.0e-4, 1.0],
                [-0.3, 0.7, -0.2],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        quaternion = edge_quaternion_reference(edge_vec)
        norm = torch.linalg.vector_norm(quaternion, dim=-1)
        self.assertTrue(torch.isfinite(quaternion).all())
        torch.testing.assert_close(
            norm, torch.ones_like(norm), atol=1.0e-6, rtol=1.0e-6
        )
        (gradient,) = torch.autograd.grad(
            quaternion.square().sum() + quaternion[:, 0].sum(), edge_vec
        )
        self.assertTrue(torch.isfinite(gradient).all())

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_edge_quaternion(torch.zeros((2, 3), dtype=torch.float32))

    def test_invalid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"shape \(n_edges, 3\)"):
            tilelang_edge_quaternion(torch.zeros((2, 4), dtype=torch.float32))

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        edge_vec = torch.randn((41, 3), dtype=torch.float32, device="cuda")
        edge_vec[0] = torch.tensor([0.0, 0.0, 1.0], device="cuda")
        edge_vec[1] = torch.tensor([0.0, 0.0, -1.0], device="cuda")
        edge_vec[2] = torch.tensor([1.0e-4, -2.0e-4, 1.0], device="cuda")
        edge_vec.requires_grad_()
        upstream = torch.randn((41, 4), dtype=torch.float32, device="cuda")

        actual = tilelang_edge_quaternion(edge_vec)
        actual_grad = torch.autograd.grad(actual, edge_vec, upstream)[0]
        reference_vec = edge_vec.detach().clone().requires_grad_(True)
        expected = edge_quaternion_reference(reference_vec)
        expected_grad = torch.autograd.grad(expected, reference_vec, upstream)[0]
        torch.testing.assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-6)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=8.0e-4, atol=4.0e-5)

        cached_vec = edge_vec.detach().clone().requires_grad_(True)
        edge_len = torch.sqrt(
            (edge_vec.detach() * edge_vec.detach()).sum(-1, keepdim=True) + 1e-14
        ).requires_grad_(True)
        cached_actual = tilelang_edge_quaternion(cached_vec, edge_len=edge_len)
        cached_grads = torch.autograd.grad(
            cached_actual, (cached_vec, edge_len), upstream
        )
        reference_vec = edge_vec.detach().clone().requires_grad_(True)
        reference_len = (
            torch.sqrt((reference_vec * reference_vec).sum(-1, keepdim=True) + 1e-14)
            .detach()
            .requires_grad_(True)
        )
        cached_expected = edge_quaternion_reference(
            reference_vec, edge_len=reference_len
        )
        expected_grads = torch.autograd.grad(
            cached_expected, (reference_vec, reference_len), upstream
        )
        torch.testing.assert_close(
            cached_actual, cached_expected, rtol=3.0e-5, atol=3.0e-6
        )
        for actual_grad, expected_grad in zip(cached_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=8.0e-4, atol=4.0e-5
            )


if __name__ == "__main__":
    unittest.main()
