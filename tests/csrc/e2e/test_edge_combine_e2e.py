"""End-to-end test for the C6 edge-combine operator (operator scope only).

This is not the model-level ladder: it runs the operator as it is used inside a
real block -- a full-size edge stream, a forward chain, the backward pass through
that chain, a downstream consumer, and a non-contiguous factor -- and compares
every result against the equivalent eager PyTorch chain.

Run on a supported HIP/DTK DCU with::

    DPA4_RUN_EDGE_COMBINE_E2E=1 PYTHONPATH=python \\
        python -m unittest discover -s tests/csrc/e2e -v

``DPA4_EDGE_E2E_EDGES`` overrides the edge count (default: the profiled 181838).
"""
from __future__ import annotations

import os
import unittest

import torch

from kernels.combine import edge_combine, edge_combine_available

_RUN = os.environ.get("DPA4_RUN_EDGE_COMBINE_E2E", "0").lower() in {
    "1", "true", "yes", "on"
}
_EDGES = int(os.environ.get("DPA4_EDGE_E2E_EDGES", "181838"))
_ROWS = int(os.environ.get("DPA4_EDGE_E2E_ROWS", "9"))
_CHANNELS = 64


@unittest.skipUnless(
    _RUN and edge_combine_available() and torch.cuda.is_available(),
    "set DPA4_RUN_EDGE_COMBINE_E2E=1 on a built DCU host",
)
class EdgeCombineEndToEndTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cuda").manual_seed(13)
        self.edges, self.rows, self.channels = _EDGES, _ROWS, _CHANNELS
        self.value = torch.randn(
            self.edges, self.rows, self.channels, device="cuda", generator=generator
        )
        self.residual = torch.randn_like(self.value)
        self.gate = torch.randn(
            self.edges, self.rows, 1, device="cuda", generator=generator
        )
        self.upstream = torch.randn_like(self.value)

    def tearDown(self):
        torch.cuda.empty_cache()

    def test_forward_chain_matches_the_eager_chain(self):
        """(x + x * gate + residual) on the full edge stream."""
        fused = edge_combine(self.value, (self.value, self.gate), self.residual)
        eager = self.value + self.value * self.gate + self.residual
        self.assertEqual(tuple(fused.shape), tuple(eager.shape))
        torch.testing.assert_close(fused, eager, rtol=1e-6, atol=1e-6)

    def test_backward_through_the_chain_matches_eager(self):
        value = self.value.detach().clone().requires_grad_()
        gate = self.gate.detach().clone().requires_grad_()
        residual = self.residual.detach().clone().requires_grad_()
        eager_value = value.detach().clone().requires_grad_()
        eager_gate = gate.detach().clone().requires_grad_()
        eager_residual = residual.detach().clone().requires_grad_()

        fused = edge_combine(value, (value, gate), residual)
        (fused * self.upstream).sum().backward()
        eager = eager_value + eager_value * eager_gate + eager_residual
        (eager * self.upstream).sum().backward()

        for produced, expected in (
            (value, eager_value), (gate, eager_gate), (residual, eager_residual)
        ):
            self.assertIsNotNone(produced.grad)
            torch.testing.assert_close(produced.grad, expected.grad, rtol=1e-6, atol=1e-6)

    def test_chained_with_a_downstream_reduction(self):
        """The fused output feeds a reduction, and gradients cross both ops."""
        value = self.value.detach().clone().requires_grad_()
        gate = self.gate.detach().clone().requires_grad_()
        residual = self.residual.detach().clone().requires_grad_()
        eager_value = value.detach().clone().requires_grad_()
        eager_gate = gate.detach().clone().requires_grad_()
        eager_residual = residual.detach().clone().requires_grad_()
        head = torch.randn(self.rows, self.channels, device="cuda")

        fused = (edge_combine(value, (value, gate), residual) * head).sum(dim=(1, 2)).sum()
        fused.backward()
        eager = ((eager_value + eager_value * eager_gate + eager_residual) * head
                 ).sum(dim=(1, 2)).sum()
        eager.backward()

        for produced, expected in (
            (value, eager_value), (gate, eager_gate), (residual, eager_residual)
        ):
            torch.testing.assert_close(produced.grad, expected.grad, rtol=1e-6, atol=1e-6)

    def test_non_contiguous_factor_keeps_its_gradient(self):
        """A sliced per-channel factor is densified inside, gradients still flow."""
        generator = torch.Generator(device="cuda").manual_seed(21)
        wide = torch.randn(
            self.edges, self.rows, 2 * self.channels, device="cuda", generator=generator
        )
        factor = wide[..., ::2]
        self.assertFalse(factor.is_contiguous())

        actual = edge_combine(self.value, (self.value, factor))
        expected = self.value + self.value * factor
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

        value = self.value.detach().clone().requires_grad_()
        wide_grad = wide.detach().clone().requires_grad_()
        edge_combine(value, (value, wide_grad[..., ::2])).sum().backward()
        eager_value = self.value.detach().clone().requires_grad_()
        wide_ref = wide.detach().clone().requires_grad_()
        (eager_value + eager_value * wide_ref[..., ::2]).sum().backward()
        torch.testing.assert_close(value.grad, eager_value.grad, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(wide_grad.grad, wide_ref.grad, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
