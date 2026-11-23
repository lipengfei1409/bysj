"""PyTorch reference for the fused edge-stream elementwise chain (C6).

``edge_combine`` replaces a chain of broadcast elementwise kernels with one
pass; the reference is written with plain broadcasting so the contract is
readable. A term is a tensor (linear) or a tuple of tensors (product).
"""
from __future__ import annotations

import torch


def _term_factors(term):
    if isinstance(term, torch.Tensor):
        return (term,)
    return tuple(term)


def edge_combine_reference(terms, coefficients=None):
    """``out = sum_t coef_t * prod_f factor_{t,f}`` with plain broadcasting."""
    if not terms:
        raise ValueError("edge combine needs at least one term")
    coefficients = (
        [1.0] * len(terms)
        if coefficients is None
        else [float(value) for value in coefficients]
    )
    if len(coefficients) != len(terms):
        raise ValueError("edge combine coefficients must match the term count")
    out = None
    for term, coefficient in zip(terms, coefficients):
        factors = _term_factors(term)
        product = factors[0]
        for factor in factors[1:]:
            product = product * factor
        piece = product if coefficient == 1.0 else product * coefficient
        out = piece if out is None else out + piece
    return out


__all__ = ["edge_combine_reference"]
