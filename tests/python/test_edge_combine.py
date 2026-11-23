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
    def edge_combine(factors, modes, coefficients, term_starts, term_counts,
                     edges, rows, channels):
        terms = [
            tuple(factors[start:start + count])
            for start, count in zip(term_starts, term_counts)
        ]
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

    def test_product_term_forward_and_gradients(self):
        generator = torch.Generator().manual_seed(7)
        gate = torch.randn(5, 3, 1, generator=generator)
        bias = torch.randn(5, 1, 8, generator=generator)
        value = torch.randn(5, 3, 8, generator=generator)
        terms = (value, (value, gate), bias)
        coefficients = (1.0, 2.0, 0.5)
        actual = combine.edge_combine(*terms, coefficients=coefficients)
        expected = edge_combine_reference(terms, coefficients)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Clone each factor, keeping the term structure (the product is a tuple).
        actual = [t.detach().clone().requires_grad_() for t in (value, gate, bias)]
        expected = [t.detach().clone().requires_grad_() for t in (value, gate, bias)]
        combine.edge_combine(
            actual[0], (actual[0], actual[1]), actual[2], coefficients=coefficients
        ).sum().backward()
        edge_combine_reference(
            (expected[0], (expected[0], expected[1]), expected[2]), coefficients
        ).sum().backward()
        for produced, reference in zip(actual, expected):
            torch.testing.assert_close(produced.grad, reference.grad, rtol=1e-6, atol=1e-6)
        # d/dgate is the coefficient times the other factor, summed over C
        torch.testing.assert_close(
            actual[1].grad, (2.0 * value).sum(dim=2, keepdim=True)
        )

    def test_row_and_channel_constant_broadcast(self):
        full = torch.randn(4, 3, 8)
        row_const = torch.randn(1, 3, 1)
        channel_const = torch.randn(1, 1, 8)
        self.assertEqual(combine._mode_of(row_const, 4, 3, 8), combine.ROW_CONST)
        self.assertEqual(combine._mode_of(channel_const, 4, 3, 8), combine.CHANNEL_CONST)
        actual = combine.edge_combine(full, row_const, channel_const)
        expected = edge_combine_reference((full, row_const, channel_const))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_five_dimensional_singletons_are_squeezed(self):
        factor = torch.randn(5, 4, 1, 1, 8)
        squeezed = combine._normalize(factor)
        self.assertEqual(tuple(squeezed.shape), (5, 4, 8))
        edge_only = torch.randn(5, 1, 1, 1, 1)
        self.assertEqual(tuple(combine._normalize(edge_only).shape), (5, 1, 1))
        two_d = torch.randn(5, 4)
        self.assertEqual(tuple(combine._normalize(two_d).shape), (5, 4, 1))

    def test_rejects_too_many_factors(self):
        full = torch.randn(4, 3, 8)
        with self.assertRaises(ValueError):
            combine.edge_combine(*(full for _ in range(9)))

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
