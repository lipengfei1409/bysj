"""Tests for the standalone TileLang S2 grid pointwise product."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.grid_product import (
    grid_product_reference,
    tilelang_grid_product,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


class GridProductTests(unittest.TestCase):
    def test_reference_forward_and_gradients(self):
        left = torch.randn((2, 7, 3, 8), dtype=torch.float64, requires_grad=True)
        right = torch.randn_like(left, requires_grad=True)
        output = grid_product_reference(left, right)
        expected = left * right
        torch.testing.assert_close(output, expected)
        upstream = torch.randn_like(output)
        grads = torch.autograd.grad(output, (left, right), upstream)
        expected_grads = torch.autograd.grad(expected, (left, right), upstream)
        for grad, expected_grad in zip(grads, expected_grads):
            torch.testing.assert_close(grad, expected_grad)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        value = torch.ones((2, 4, 3), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_grid_product(value, value)

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        left = torch.randn(
            (2, 31, 1, 64), dtype=torch.float32, device="cuda", requires_grad=True
        )
        right = torch.randn_like(left, requires_grad=True)
        upstream = torch.randn_like(left)
        actual = tilelang_grid_product(left, right)
        actual_grads = torch.autograd.grad(actual, (left, right), upstream)
        ref_left = left.detach().clone().requires_grad_(True)
        ref_right = right.detach().clone().requires_grad_(True)
        expected = grid_product_reference(ref_left, ref_right)
        expected_grads = torch.autograd.grad(expected, (ref_left, ref_right), upstream)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        for grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(grad, expected_grad, rtol=2.0e-5, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
