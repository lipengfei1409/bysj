"""PyTorch reference for the fused edge-stream linear combination (C6).

``edge_combine`` replaces a chain of broadcast elementwise kernels with one
pass; the reference is written with plain broadcasting so the contract is
readable.
"""
from __future__ import annotations

import torch


def edge_combine_reference(terms, coefficients=None):
    """``out = sum_t coef_t * term_t`` with the same broadcast rules."""
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
        piece = term.reshape(1, 1, 1) * coefficient if term.dim() <= 1 else term * coefficient
        out = piece if out is None else out + piece
    return out


__all__ = ["edge_combine_reference"]
