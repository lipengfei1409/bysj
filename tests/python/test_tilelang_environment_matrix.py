"""Tests for the standalone TileLang DPA4 environment-matrix operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops import (
    environment_matrix_reference,
    tilelang_environment_matrix,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class EnvironmentMatrixTests(unittest.TestCase):
    def test_reference_matches_formula_and_gradients(self):
        edge_vec = torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, -1.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        edge_env = torch.tensor([[0.0], [0.7]], dtype=torch.float64, requires_grad=True)
        actual = environment_matrix_reference(edge_vec, edge_env)
        inv_r = torch.rsqrt((edge_vec * edge_vec).sum(-1, keepdim=True) + 1e-14)
        scale = edge_env * inv_r
        expected = torch.cat((scale, scale * edge_vec * inv_r), dim=-1)
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, (edge_vec, edge_env), upstream)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        vec = torch.zeros((2, 3), dtype=torch.float32)
        env = torch.ones((2, 1), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_environment_matrix(vec, env)

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"shape \(n_edges, 3\)"):
            tilelang_environment_matrix(torch.zeros((2, 4)), torch.ones((2, 1)))
        with self.assertRaisesRegex(ValueError, r"shape \(n_edges, 1\)"):
            tilelang_environment_matrix(torch.zeros((2, 3)), torch.ones((2,)))

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        edge_vec = torch.randn((31, 3), dtype=torch.float32, device="cuda")
        edge_vec[0] = 0.0
        edge_vec.requires_grad_()
        edge_env = torch.rand((31, 1), dtype=torch.float32, device="cuda")
        edge_env.requires_grad_()
        upstream = torch.randn((31, 4), dtype=torch.float32, device="cuda")
        actual = tilelang_environment_matrix(edge_vec, edge_env)
        actual_grads = torch.autograd.grad(actual, (edge_vec, edge_env), upstream)

        ref_vec = edge_vec.detach().clone().requires_grad_(True)
        ref_env = edge_env.detach().clone().requires_grad_(True)
        expected = environment_matrix_reference(ref_vec, ref_env)
        expected_grads = torch.autograd.grad(expected, (ref_vec, ref_env), upstream)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-4, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
