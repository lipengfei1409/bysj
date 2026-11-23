"""Opt-in accuracy test for the C6 edge-combine operator on a DCU."""
import unittest

import torch

from kernels.combine import edge_combine, edge_combine_available
from reference.combine import edge_combine_reference


@unittest.skipUnless(
    edge_combine_available() and torch.cuda.is_available(),
    "requires a built edge-combine HIP extension and a supported DCU",
)
class EdgeCombineAccuracyTests(unittest.TestCase):
    def _terms(self, edges, rows, channels, seed=3):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return {
            "full": torch.randn(edges, rows, channels, device="cuda", generator=generator),
            "per_row": torch.randn(edges, rows, 1, device="cuda", generator=generator),
            "per_channel": torch.randn(edges, 1, channels, device="cuda", generator=generator),
            "per_edge": torch.randn(edges, 1, 1, device="cuda", generator=generator),
            "scalar": torch.randn(1, device="cuda", generator=generator),
        }

    def test_matches_the_reference_for_every_broadcast_mix(self):
        for edges, rows, channels in ((7, 3, 8), (5, 10, 64), (1, 4, 16)):
            with self.subTest(shape=(edges, rows, channels)):
                terms = self._terms(edges, rows, channels)
                combos = (
                    (("full",), (1.0,)),
                    (("full", "per_row", "per_channel", "scalar"), (1.0, 0.5, -2.0, 3.0)),
                    (("per_row", "per_edge"), (1.0, -1.0)),
                    (("full", "scalar"), (0.25, 2.0)),
                )
                for names, coefficients in combos:
                    with self.subTest(names=names):
                        actual = edge_combine(
                            *(terms[name] for name in names), coefficients=coefficients
                        )
                        expected = edge_combine_reference(
                            tuple(terms[name] for name in names), coefficients
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gradients_match_the_reference(self):
        edges, rows, channels = 6, 4, 16
        terms = self._terms(edges, rows, channels, seed=9)
        names = ("full", "per_row", "per_channel", "scalar")
        coefficients = (1.0, 0.5, -2.0, 3.0)
        actual_terms = [terms[name].detach().clone().requires_grad_() for name in names]
        expected_terms = [terms[name].detach().clone().requires_grad_() for name in names]
        edge_combine(*actual_terms, coefficients=coefficients).sum().backward()
        edge_combine_reference(expected_terms, coefficients).sum().backward()
        for actual, expected in zip(actual_terms, expected_terms):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-6, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
