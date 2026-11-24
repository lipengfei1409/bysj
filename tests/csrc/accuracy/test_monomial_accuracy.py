"""Opt-in accuracy tests for the fused monomial operators on a DCU."""
import unittest

import torch

from kernels.monomial import (
    monomial_runs,
    monomial_runs_available,
    monomial_runs_backward,
)
from reference.monomial import (
    monomial_runs_backward_reference,
    monomial_runs_reference,
)


def _tables(degree: int):
    """Every quaternion monomial up to ``degree``, and a run table over them."""
    exps = []
    for total in range(degree + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                for c in range(total - a - b + 1):
                    exps.append([a, b, c, total - a - b - c])
    table = torch.tensor(exps, dtype=torch.int8)
    return table, table.flip(1).to(torch.int8)


@unittest.skipUnless(
    monomial_runs_available() and torch.cuda.is_available(),
    "requires a built monomial HIP extension and a supported DCU",
)
class MonomialAccuracyTests(unittest.TestCase):
    device = "cuda"

    def _quat(self, edges):
        q = torch.randn(edges, 4, device=self.device)
        return q / q.norm(dim=1, keepdim=True)

    def test_runs_match_the_explicit_design_matrix(self):
        for degree, n_runs in ((2, 12), (4, 30), (6, 46)):
            with self.subTest(degree=degree):
                exps, dexp = _tables(degree)
                exps = exps.to(self.device)
                coeff = torch.randn(n_runs, exps.shape[0], device=self.device)
                q = self._quat(257)
                torch.testing.assert_close(
                    monomial_runs(q, exps, coeff),
                    monomial_runs_reference(q, exps, coeff),
                    rtol=2e-5,
                    atol=2e-6,
                )

    def test_an_identity_run_table_picks_single_monomials(self):
        exps, _ = _tables(2)
        exps = exps.to(self.device)
        idx = torch.tensor([0, 1, 5, 9], dtype=torch.int64)
        coeff = torch.zeros(4, exps.shape[0], device=self.device)
        coeff[torch.arange(4), idx] = 1.0
        q = self._quat(64)
        actual = monomial_runs(q, exps, coeff)
        expected = monomial_runs_reference(q, exps, coeff)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_backward_matches_the_contraction(self):
        exps, dexp = _tables(3)
        exps, dexp = exps.to(self.device), dexp.to(self.device)
        n_runs = 20
        coeff = torch.randn(n_runs, exps.shape[0], device=self.device)
        dcoeff = torch.randn(n_runs, 4, dexp.shape[0], device=self.device)
        q = self._quat(129)
        grad_runs = torch.randn(129, n_runs, device=self.device)
        torch.testing.assert_close(
            monomial_runs_backward(grad_runs, q, dexp, dcoeff),
            monomial_runs_backward_reference(grad_runs, q, dexp, dcoeff),
            rtol=2e-5,
            atol=2e-6,
        )

    def test_the_result_is_deterministic(self):
        exps, _ = _tables(4)
        exps = exps.to(self.device)
        coeff = torch.randn(24, exps.shape[0], device=self.device)
        q = self._quat(512)
        first = monomial_runs(q, exps, coeff)
        second = monomial_runs(q, exps, coeff)
        torch.testing.assert_close(first, second, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
