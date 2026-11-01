"""Tests for the standalone TileLang l=1 Wigner-D operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.wigner_l1 import tilelang_wigner_l1, wigner_l1_reference

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class WignerL1Tests(unittest.TestCase):
    def test_reference_is_orthogonal_and_has_finite_quaternion_gradient(self):
        quaternion = torch.randn((8, 4), dtype=torch.float64, requires_grad=True)
        d_full, dt_full = wigner_l1_reference(quaternion)
        identity = torch.eye(4, dtype=d_full.dtype).expand(8, 4, 4)
        torch.testing.assert_close(
            d_full @ dt_full, identity, rtol=1.0e-12, atol=1.0e-12
        )
        torch.testing.assert_close(dt_full, d_full.transpose(-1, -2))

        loss = (d_full.square() * torch.randn_like(d_full)).sum()
        (grad,) = torch.autograd.grad(loss, quaternion)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().max().item(), 0.0)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_wigner_l1(torch.zeros((2, 4), dtype=torch.float32))

    def test_invalid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"shape \(n_edges, 4\)"):
            tilelang_wigner_l1(torch.zeros((2, 3), dtype=torch.float32))

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        quaternion = torch.randn(
            (31, 4), dtype=torch.float32, device="cuda", requires_grad=True
        )
        grad_d = torch.randn((31, 4, 4), dtype=torch.float32, device="cuda")
        grad_dt = torch.randn_like(grad_d)
        actual_d, actual_dt = tilelang_wigner_l1(quaternion)
        (actual_grad,) = torch.autograd.grad(
            (actual_d, actual_dt), quaternion, (grad_d, grad_dt)
        )

        reference_q = quaternion.detach().clone().requires_grad_(True)
        expected_d, expected_dt = wigner_l1_reference(reference_q)
        (expected_grad,) = torch.autograd.grad(
            (expected_d, expected_dt), reference_q, (grad_d, grad_dt)
        )
        torch.testing.assert_close(actual_d, expected_d, rtol=2.0e-5, atol=2.0e-6)
        torch.testing.assert_close(actual_dt, expected_dt, rtol=2.0e-5, atol=2.0e-6)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3.0e-4, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
