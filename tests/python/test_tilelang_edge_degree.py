"""Tests for the standalone TileLang DPA4 edge degree operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops import edge_degree_reference, tilelang_edge_degree

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class EdgeDegreeTests(unittest.TestCase):
    def test_reference_matches_segmented_degree_and_gradients(self):
        dst = torch.tensor([1, 0, 1, 3, 1], dtype=torch.int64)
        edge_env = torch.tensor(
            [[0.4], [0.0], [0.8], [1.0], [0.25]],
            dtype=torch.float64,
            requires_grad=True,
        )
        deg, inv_sqrt_deg = edge_degree_reference(
            edge_env, dst, n_nodes=4, deg_norm_floor=0.25
        )
        expected_deg = torch.tensor([0.0, 0.8625, 0.0, 1.0], dtype=torch.float64)
        torch.testing.assert_close(deg, expected_deg)
        torch.testing.assert_close(
            inv_sqrt_deg.reshape(-1), torch.rsqrt(expected_deg + 0.25)
        )
        grad_deg = torch.randn_like(deg)
        grad_inv = torch.randn_like(inv_sqrt_deg)
        grads = torch.autograd.grad((deg, inv_sqrt_deg), edge_env, (grad_deg, grad_inv))
        self.assertTrue(torch.isfinite(grads[0]).all())
        self.assertEqual(grads[0][1].item(), 0.0)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        edge_env = torch.ones((3, 1), dtype=torch.float32)
        dst = torch.tensor([0, 1, 0], dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_edge_degree(edge_env, dst, n_nodes=2, deg_norm_floor=0.25)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"shape \(n_edges, 1\)"):
            tilelang_edge_degree(
                torch.ones((3,), dtype=torch.float32),
                torch.zeros((3,), dtype=torch.int64),
                n_nodes=2,
                deg_norm_floor=0.25,
            )
        with self.assertRaisesRegex(ValueError, "deg_norm_floor"):
            tilelang_edge_degree(
                torch.ones((3, 1), dtype=torch.float32),
                torch.zeros((3,), dtype=torch.int64),
                n_nodes=2,
                deg_norm_floor=-0.1,
            )

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        dst = torch.tensor([1, 0, 1, 3, 1], dtype=torch.int64, device="cuda")
        edge_env = torch.tensor(
            [[0.4], [0.0], [0.8], [1.0], [0.25]],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        upstream_deg = torch.randn((4,), dtype=torch.float32, device="cuda")
        upstream_inv = torch.randn((4, 1, 1), dtype=torch.float32, device="cuda")
        actual = tilelang_edge_degree(edge_env, dst, 4, 0.25)
        actual_grad = torch.autograd.grad(
            actual, edge_env, (upstream_deg, upstream_inv)
        )[0]

        ref_env = edge_env.detach().clone().requires_grad_(True)
        expected = edge_degree_reference(ref_env, dst, 4, 0.25)
        expected_grad = torch.autograd.grad(
            expected, ref_env, (upstream_deg, upstream_inv)
        )[0]
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(
                actual_value, expected_value, rtol=2e-5, atol=2e-6
            )
        torch.testing.assert_close(actual_grad, expected_grad, rtol=4e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
