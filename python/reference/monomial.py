"""Explicit oracles for the C7-a monomial operators.

The oracle builds the monomial design matrix explicitly (the pass the fused
kernel removes), so the two implementations differ in exactly the way the
operator claims to improve.
"""

from __future__ import annotations

import torch


def _monomial_matrix(q: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """(E, M) design matrix: ``prod_i q[e, i] ** exps[m, i]``."""
    powers = q.to(torch.float64).unsqueeze(1)  # (E, 1, 4)
    table = exps.to(torch.float64).unsqueeze(0)  # (1, M, 4)
    return torch.prod(powers**table, dim=-1)


def monomial_runs_reference(q: torch.Tensor, exps: torch.Tensor,
                            coeff: torch.Tensor) -> torch.Tensor:
    """``runs[e, nw] = sum_m M[e, m] * coeff[nw, m]``, in float64."""
    design = _monomial_matrix(q, exps)
    return (design @ coeff.to(torch.float64).t()).to(q.dtype)


def monomial_runs_backward_reference(grad_runs: torch.Tensor, q: torch.Tensor,
                                     dexp: torch.Tensor,
                                     dcoeff: torch.Tensor) -> torch.Tensor:
    """``g_quat[e, i] = sum_m' dM[e, m'] * sum_nw g_runs[e, nw] * dcoeff[nw, i, m']``."""
    design = _monomial_matrix(q, dexp)  # (E, M')
    partial = torch.einsum("en,nim->eim", grad_runs.to(torch.float64),
                           dcoeff.to(torch.float64))
    return torch.einsum("eim,em->ei", partial, design).to(q.dtype)
