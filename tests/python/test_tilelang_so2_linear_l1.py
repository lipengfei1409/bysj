"""Tests for the standalone TileLang DPA4 lmax=1 SO2Linear operator."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.so2_linear_l1 import (
    so2_linear_l1_reference,
    tilelang_so2_linear_l1,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


def _make_inputs(device="cpu", requires_grad=False):
    x = torch.randn((2, 5, 8), dtype=torch.float32, device=device)
    weight = torch.zeros((8, 2, 12), dtype=torch.float32, device=device)
    weight[:4, :, :6] = torch.randn((4, 2, 6), dtype=torch.float32, device=device)
    weight[4:, :, 6:] = torch.randn((4, 2, 6), dtype=torch.float32, device=device)
    if requires_grad:
        x.requires_grad_()
    return x, weight


class SO2LinearL1Tests(unittest.TestCase):
    def test_reference_matches_independent_block_matmuls_and_gradient(self):
        x, weight = _make_inputs(requires_grad=True)
        actual = so2_linear_l1_reference(x, weight)
        expected = torch.cat(
            [
                x[..., :4] @ weight[:4, :, :6].permute(1, 0, 2),
                x[..., 4:] @ weight[4:, :, 6:].permute(1, 0, 2),
            ],
            dim=-1,
        )
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        (actual_grad,) = torch.autograd.grad(actual, x, upstream)
        (expected_grad,) = torch.autograd.grad(expected, x, upstream)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        x, weight = _make_inputs()
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_so2_linear_l1(x, weight)

    def test_invalid_layout_is_rejected(self):
        x = torch.randn((1, 2, 5), dtype=torch.float32)
        weight = torch.randn((5, 1, 8), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, r"two equal SO\(2\) blocks"):
            tilelang_so2_linear_l1(x, weight)

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        x, weight = _make_inputs(device="cuda", requires_grad=True)
        upstream = torch.randn((2, 5, 12), dtype=torch.float32, device="cuda")
        actual = tilelang_so2_linear_l1(x, weight)
        (actual_grad,) = torch.autograd.grad(actual, x, upstream)
        ref_x = x.detach().clone().requires_grad_(True)
        expected = so2_linear_l1_reference(ref_x, weight)
        (expected_grad,) = torch.autograd.grad(expected, ref_x, upstream)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=2.0e-4, atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()
