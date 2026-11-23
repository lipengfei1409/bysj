"""Host tests for the C6 edge-combine wrapper and its autograd.

The native kernel needs a DCU; everything above the extension call - shape
inference, broadcast modes, coefficient handling and the backward reductions -
runs on the host against the reference implementation.
"""
from __future__ import annotations

import unittest

import torch

from kernels import combine
from reference.combine import edge_combine_reference


class _FakeExtension:
    @staticmethod
    def edge_combine(terms, coefficients, modes, edges, rows, channels):
        return edge_combine_reference(terms, coefficients)


def _fake_extension(name):
    return _FakeExtension()


class EdgeCombineHostTests(unittest.TestCase):
    def setUp(self):
        self._original_extension = combine._extension
        self._original_require = combine.require_dcu
        combine._extension = _fake_extension
        combine.require_dcu = lambda *tensors: None
        self.addCleanup(setattr, combine, "_extension", self._original_extension)
        self.addCleanup(setattr, combine, "require_dcu", self._original_require)

    @staticmethod
    def _inputs(edges=5, rows=3, channels=8):
        generator = torch.Generator().manual_seed(11)
        return (
            torch.randn(edges, rows, channels, generator=generator),
            torch.randn(edges, rows, 1, generator=generator),
            torch.randn(edges, 1, channels, generator=generator),
            torch.randn(1, generator=generator),
        )

    def test_forward_matches_the_reference(self):
        full, per_row, per_channel, scalar = self._inputs()
        actual = combine.edge_combine(
            full, per_row, per_channel, scalar,
            coefficients=(1.0, 0.5, -2.0, 3.0),
        )
        expected = edge_combine_reference(
            (full, per_row, per_channel, scalar), (1.0, 0.5, -2.0, 3.0)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_broadcast_modes_are_inferred(self):
        full, per_row, per_channel, _ = self._inputs()
        self.assertEqual(combine._shape_of((full, per_row, per_channel)), (5, 3, 8))
        self.assertEqual(combine._mode_of(full, 5, 3, 8), combine.FULL)
        self.assertEqual(combine._mode_of(per_row, 5, 3, 8), combine.PER_ROW)
        self.assertEqual(combine._mode_of(per_channel, 5, 3, 8), combine.PER_CHANNEL)
        per_edge = torch.randn(5, 1, 1)
        self.assertEqual(combine._mode_of(per_edge, 5, 3, 8), combine.PER_EDGE)

    def test_gradients_match_the_reference(self):
        full, per_row, per_channel, scalar = self._inputs()
        terms = [t.clone().requires_grad_() for t in (full, per_row, per_channel, scalar)]
        coefficients = (1.0, 0.5, -2.0, 3.0)
        combine.edge_combine(*terms, coefficients=coefficients).sum().backward()

        reference_terms = [
            t.detach().clone().requires_grad_() for t in (full, per_row, per_channel, scalar)
        ]
        edge_combine_reference(reference_terms, coefficients).sum().backward()

        for produced, expected, term in zip(terms, reference_terms, terms):
            self.assertEqual(tuple(produced.grad.shape), tuple(term.shape))
            torch.testing.assert_close(produced.grad, expected.grad, rtol=1e-6, atol=1e-6)

    def test_gradient_of_a_broadcast_term_is_reduced(self):
        full = torch.randn(4, 3, 8)
        per_row = torch.randn(4, 3, 1, requires_grad=True)
        scalar = torch.tensor(2.0, requires_grad=True)
        combine.edge_combine(full, per_row, scalar).sum().backward()
        torch.testing.assert_close(per_row.grad, torch.full((4, 3, 1), 8.0))
        torch.testing.assert_close(scalar.grad, torch.tensor(96.0))

    def test_rejects_terms_that_do_not_broadcast(self):
        full = torch.randn(4, 3, 8)
        with self.assertRaises(ValueError):
            combine.edge_combine(full, torch.randn(4, 2, 8))
        with self.assertRaises(ValueError):
            combine.edge_combine(torch.randn(4, 3, 6))  # channels not a multiple of 4
        with self.assertRaises(ValueError):
            combine.edge_combine(full, full, full, full, full)  # more than four


if __name__ == "__main__":
    unittest.main()
