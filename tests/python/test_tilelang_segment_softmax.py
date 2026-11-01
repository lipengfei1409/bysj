"""Tests for the standalone TileLang DPA4 segment attention softmax."""

import importlib.util
import unittest

import torch
from kernels.tilelang_ops.segment_softmax import (
    segment_envelope_softmax_reference,
    tilelang_segment_envelope_softmax,
)

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)


def _topology(device="cpu"):
    dst = torch.tensor([1, 0, 1, 3, 1], dtype=torch.int64, device=device)
    order = torch.argsort(dst, stable=True)
    counts = torch.bincount(dst, minlength=4)
    row_ptr = torch.cat(
        [torch.zeros((1,), dtype=torch.int64, device=device), counts.cumsum(0)]
    )
    return dst, order, row_ptr


class SegmentSoftmaxTests(unittest.TestCase):
    def test_reference_matches_manual_segment_formula_and_gradients(self):
        dst, _, _ = _topology()
        logits = torch.randn((5, 2, 2), dtype=torch.float64, requires_grad=True)
        edge_env = torch.tensor(
            [0.4, 0.0, 0.8, 1.0, 0.25], dtype=torch.float64, requires_grad=True
        )
        null_logit = torch.randn((2, 2), dtype=torch.float64)
        actual = segment_envelope_softmax_reference(
            logits, edge_env, dst, 4, null_logit
        )

        expected = torch.zeros_like(logits)
        for node in range(4):
            edges = torch.nonzero(dst == node).flatten()
            for focus in range(2):
                for head in range(2):
                    mass = edge_env[edges].square() * logits.new_ones((len(edges),))
                    numerator = mass * torch.exp(logits[edges, focus, head])
                    denominator = numerator.sum() + torch.exp(null_logit[focus, head])
                    expected[edges, focus, head] = numerator / denominator
        torch.testing.assert_close(actual, expected)
        upstream = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, (logits, edge_env), upstream)
        self.assertTrue(torch.isfinite(grads[0]).all())
        self.assertTrue(torch.isfinite(grads[1]).all())
        self.assertEqual(grads[1][1].item(), 0.0)

    def test_tilelang_op_does_not_silently_fall_back_on_cpu(self):
        dst, order, row_ptr = _topology()
        logits = torch.zeros((5, 1, 1), dtype=torch.float32)
        env = torch.ones((5,), dtype=torch.float32)
        null_logit = torch.zeros((1, 1), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "requires a HIP/DCU tensor"):
            tilelang_segment_envelope_softmax(
                logits, env, dst, order, row_ptr, null_logit
            )

    @unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
    def test_forward_and_backward_match_reference(self):
        dst, order, row_ptr = _topology("cuda")
        logits = torch.randn(
            (5, 2, 2), dtype=torch.float32, device="cuda", requires_grad=True
        )
        edge_env = torch.tensor(
            [0.4, 0.0, 0.8, 1.0, 0.25],
            dtype=torch.float32,
            device="cuda",
            requires_grad=True,
        )
        null_logit = torch.randn((2, 2), dtype=torch.float32, device="cuda")
        upstream = torch.randn_like(logits)
        actual = tilelang_segment_envelope_softmax(
            logits, edge_env, dst, order, row_ptr, null_logit
        )
        actual_grads = torch.autograd.grad(actual, (logits, edge_env), upstream)

        ref_logits = logits.detach().clone().requires_grad_(True)
        ref_env = edge_env.detach().clone().requires_grad_(True)
        expected = segment_envelope_softmax_reference(
            ref_logits, ref_env, dst, 4, null_logit
        )
        expected_grads = torch.autograd.grad(expected, (ref_logits, ref_env), upstream)
        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(
                actual_grad, expected_grad, rtol=3.0e-4, atol=2.0e-5
            )


if __name__ == "__main__":
    unittest.main()
