"""Fused quaternion monomial runs (C7-a).

``monomial_runs`` replaces the dense chain "power table -> gather -> product ->
bmm" with one pass: the monomials are evaluated in registers from an exponent
table and contracted against the run coefficients. ``monomial_runs_backward``
replaces the differentiated product tree (which lowers to a ``cumprod`` scan
pair) with one contraction against the derivative tables.

Both are inference/Triton free: the exponent and coefficient tables are inputs,
so a different degree is a different table rather than a different kernel.
"""

from __future__ import annotations

import torch

from .native import _extension, require_dcu


def _validate_table(table: torch.Tensor, shape, name: str) -> None:
    if not isinstance(table, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if table.dtype != torch.int8:
        raise TypeError(f"{name} must be int8")
    if table.dim() != 2 or table.shape[1] != 4:
        raise ValueError(f"{name} must be (M, 4)")
    if not table.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def monomial_runs_available() -> bool:
    """Return whether this process can load both monomial entry points."""
    try:
        require_dcu()
        extension = _extension("monomial_runs")
    except (ImportError, OSError, RuntimeError):
        return False
    return hasattr(extension, "monomial_runs_backward")


def monomial_runs(q: torch.Tensor, exps: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """``runs[e, nw] = sum_m prod_i q[e,i]**exps[m,i] * coeff[nw,m]``."""
    if q.dim() != 2 or q.shape[1] != 4:
        raise ValueError("q must be (E, 4)")
    if q.dtype != torch.float32:
        raise TypeError("q must be float32")
    _validate_table(exps, None, "exps")
    if coeff.dtype != torch.float32 or coeff.dim() != 2:
        raise TypeError("coeff must be a float32 (NW, M) tensor")
    if coeff.shape[1] != exps.shape[0]:
        raise ValueError("coeff must have one column per monomial")
    require_dcu(q)
    return _extension("monomial_runs").monomial_runs(q, exps, coeff)


def monomial_runs_backward(grad_runs: torch.Tensor, q: torch.Tensor,
                           dexp: torch.Tensor, dcoeff: torch.Tensor) -> torch.Tensor:
    """``g_quat[e,i] = sum_m' dM[e,m'] * sum_nw g_runs[e,nw] * dcoeff[nw,i,m']``."""
    if grad_runs.dtype != torch.float32 or grad_runs.dim() != 2:
        raise TypeError("grad_runs must be a float32 (E, NW) tensor")
    _validate_table(dexp, None, "dexp")
    if dcoeff.dtype != torch.float32 or dcoeff.dim() != 3:
        raise TypeError("dcoeff must be a float32 (NW, 4, M') tensor")
    if dcoeff.shape[0] != grad_runs.shape[1] or dcoeff.shape[1] != 4:
        raise ValueError("dcoeff must be (NW, 4, M')")
    if dcoeff.shape[2] != dexp.shape[0]:
        raise ValueError("dcoeff must have one slot per derivative monomial")
    require_dcu(grad_runs, q)
    return _extension("monomial_runs_backward").monomial_runs_backward(
        grad_runs, q, dexp, dcoeff
    )
