"""Fused edge-stream linear combinations (C6).

The profiled forward and backward spend 488 ms on E-major chains of the shape
``[1, E, R, C]`` -- a residual add, a broadcast multiply, a copy and a zero fill,
each a separate pass over the same bytes. Every pass already runs at the memory
roofline, so the saving comes only from the number of passes.

``edge_combine(terms, coefficients)`` evaluates

    out[e, r, c] = sum_t coef_t * term_t[e, ...]

in one pass, one kernel and one output write. Each term broadcasts:

    (E, R, C)  full         (E, R, 1)  per row
    (E, 1, C)  per channel  (E, 1, 1)  per edge     (1,)  scalar

so ``x + residual``, ``x * gate`` and ``x + bias`` can be one call::

    edge_combine((x, residual, gate_term, bias),
                 (1.0, 1.0, -1.0, 0.5))

The op is inference-friendly and differentiable: each input's gradient is its
coefficient times the incoming gradient, reduced over the axes it broadcast.
"""
from __future__ import annotations

import torch

from .native import _extension, require_dcu

FULL = 0
PER_ROW = 1
PER_CHANNEL = 2
PER_EDGE = 3
SCALAR = 4

# The autograd wrapper has fixed positional slots; the profiled chains use at
# most four terms, and the kernel itself supports eight.
_MAX_TERMS = 4


def _shape_of(terms):
    edges = rows = channels = 1
    for term in terms:
        if term.dim() <= 1:
            if term.numel() != 1:
                raise ValueError(
                    f"edge combine 1D terms must be scalars, got {tuple(term.shape)}"
                )
            continue
        if term.dim() != 3:
            raise ValueError(
                f"edge combine terms must be 3D or scalar, got {tuple(term.shape)}"
            )
        edges = max(edges, int(term.shape[0]))
        rows = max(rows, int(term.shape[1]))
        channels = max(channels, int(term.shape[2]))
    if edges == 1 and rows == 1 and channels == 1:
        raise ValueError("edge combine needs at least one non-scalar term")
    return edges, rows, channels


def _mode_of(term, edges, rows, channels):
    if term.numel() == 1 and term.dim() <= 1:
        return SCALAR
    shape = tuple(int(value) for value in term.shape)
    if shape == (edges, rows, channels):
        return FULL
    if shape == (edges, rows, 1):
        return PER_ROW
    if shape == (edges, 1, channels):
        return PER_CHANNEL
    if shape == (edges, 1, 1):
        return PER_EDGE
    raise ValueError(
        f"edge combine term {shape} does not broadcast to ({edges}, {rows}, {channels})"
    )


def _needs_grad(terms):
    return torch.is_grad_enabled() and any(term.requires_grad for term in terms)


class _EdgeCombineFunction(torch.autograd.Function):
    """Autograd wrapper: each gradient is coef * upstream, reduced if broadcast."""

    @staticmethod
    def forward(ctx, a, b, c, d, coefficients, modes, shape):
        terms = [term for term in (a, b, c, d) if term is not None]
        edges, rows, channels = shape
        out = _extension("edge_combine").edge_combine(
            terms, list(coefficients), list(modes), edges, rows, channels
        )
        ctx.save_for_backward(a, b, c, d)
        ctx.coefficients = list(coefficients)
        ctx.modes = list(modes)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        slots = ctx.saved_tensors
        grads = [None, None, None, None]
        for slot, term in enumerate(slots):
            if term is None or not ctx.needs_input_grad[slot]:
                continue
            coefficient = ctx.coefficients[slot]
            mode = ctx.modes[slot]
            grad = grad_out if coefficient == 1.0 else grad_out * coefficient
            if mode == PER_ROW:
                grad = grad.sum(dim=2, keepdim=True)
            elif mode == PER_CHANNEL:
                grad = grad.sum(dim=1, keepdim=True)
            elif mode == PER_EDGE:
                grad = grad.sum(dim=(1, 2), keepdim=True)
            elif mode == SCALAR:
                grad = grad.sum().reshape(term.shape)
            grads[slot] = grad
        return (*grads, None, None, None)


def edge_combine(*terms, coefficients=None):
    """Return ``sum_t coef_t * term_t`` in one edge-stream pass.

    ``terms`` are contiguous float32 DCU tensors of shape ``(E, R, C)``,
    ``(E, R, 1)``, ``(E, 1, C)``, ``(E, 1, 1)`` or a scalar; ``coefficients``
    defaults to all ones. Up to four terms.
    """
    if not terms:
        raise ValueError("edge combine needs at least one term")
    if len(terms) > _MAX_TERMS:
        raise ValueError(f"edge combine supports at most {_MAX_TERMS} terms")
    for term in terms:
        if not isinstance(term, torch.Tensor):
            raise TypeError("edge combine terms must be tensors")
        if term.dtype != torch.float32:
            raise TypeError("edge combine terms must be float32")
    coefficients = (
        [1.0] * len(terms) if coefficients is None else [float(value) for value in coefficients]
    )
    if len(coefficients) != len(terms):
        raise ValueError("edge combine coefficients must match the term count")
    edges, rows, channels = _shape_of(terms)
    if channels % 4 != 0:
        raise ValueError("edge combine channels must be a multiple of 4")
    modes = [_mode_of(term, edges, rows, channels) for term in terms]
    require_dcu(*terms)
    padded = list(terms) + [None] * (_MAX_TERMS - len(terms))
    if _needs_grad(terms):
        return _EdgeCombineFunction.apply(
            *padded, coefficients, modes, (edges, rows, channels)
        )
    return _extension("edge_combine").edge_combine(
        list(terms), coefficients, modes, edges, rows, channels
    )


def edge_combine_available() -> bool:
    """Return whether this process can load the edge-combine entry point."""
    try:
        require_dcu()
        extension = _extension("edge_combine")
    except (ImportError, OSError, RuntimeError):
        return False
    return hasattr(extension, "edge_combine")


__all__ = [
    "FULL",
    "PER_ROW",
    "PER_CHANNEL",
    "PER_EDGE",
    "SCALAR",
    "edge_combine",
    "edge_combine_available",
]
