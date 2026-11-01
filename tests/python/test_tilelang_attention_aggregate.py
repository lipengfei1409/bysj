"""Tests for standalone TileLang attention-weighted message aggregation."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.attention_aggregate import (
    attention_aggregate_reference,
    tilelang_attention_aggregate,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


def _inputs(device="cpu", requires_grad=False):
    values = torch.randn((5, 4, 2, 2, 3), dtype=torch.float32, device=device)
    alpha = torch.rand((5, 2, 2), dtype=torch.float32, device=device)
    head_gate = torch.rand((4, 2, 2), dtype=torch.float32, device=device)
    dst = torch.tensor([1, 0, 1, 3, 1], dtype=torch.int64, device=device)
    if requires_grad:
        values.requires_grad_()
        alpha.requires_grad_()
        head_gate.requires_grad_()
    return values, alpha, head_gate, dst


class AttentionAggregateTests(unittest.TestCase):
    def test_reference_matches_explicit_reduction_and_gradients(self):
        inputs = _inputs(requires_grad=True)
        values, alpha, head_gate, dst = inputs
        actual = attention_aggregate_reference(*inputs)
        expected = torch.zeros((4, 4, 2, 2, 3), dtype=torch.float32)
        for edge, node in enumerate(dst.tolist()):
            expected[node] += (
                values[edge] * alpha[edge, :, :, None] * head_gate[node, :, :, None]
            )
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, (values, alpha, head_gate), upstream)
        for grad in grads:
            self.assertTrue(torch.isfinite(grad).all())

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_attention_aggregate(*_inputs())

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        inputs = _inputs(device="cuda", requires_grad=True)
        upstream = torch.randn((4, 4, 2, 2, 3), dtype=torch.float32, device="cuda")
        actual = tilelang_attention_aggregate(*inputs)
        actual_grads = torch.autograd.grad(actual, inputs[:3], upstream)

        ref_inputs = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            if value.dtype.is_floating_point
            else value
            for value in inputs
        )
        expected = attention_aggregate_reference(*ref_inputs)
        expected_grads = torch.autograd.grad(expected, ref_inputs[:3], upstream)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=3.0e-4, atol=2.0e-5
            )


if __name__ == "__main__":
    unittest.main()
