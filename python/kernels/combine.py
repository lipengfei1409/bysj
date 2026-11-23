"""Fused edge-stream elementwise chains (C6).

The profiled forward and backward spend 488 ms on E-major chains of the shape
``[1, E, R, C]`` -- a residual add, a broadcast multiply, a copy and a zero fill,
each a separate pass over the same bytes. Every pass already runs at the memory
roofline, so the saving comes only from the number of passes.

``edge_combine(terms, coefficients)`` evaluates

    out[e, r, c] = sum_t coef_t * prod_f factor_{t,f}[e, ...]

in one pass, one kernel and one output write. A term is a tensor (a linear term)
or a tuple of tensors (a product). Each factor broadcasts:

    (E, R, C)  full         (E, R, 1)  per row      (E, 1, C)  per channel
    (E, 1, 1)  per edge     (1, R, 1)  row constant  (1, 1, C)  channel constant
    (1,)       scalar

So ``(x + residual) * gate + bias`` is one call::

    edge_combine(x, residual, (gate,), bias, coefficients=(1, 1, 1, 1))

and ``x * gate`` is ``edge_combine((x, gate))``. Shapes with singleton axes above
three dimensions (``[E, 16, 1, 1, 64]``) are squeezed to the 3D broadcast form.

Each input's gradient is the coefficient times the product of the term's other
factors times the incoming gradient, reduced over the axes it broadcast.
"""
from __future__ import annotations

import torch

from .native import _extension, require_dcu

FULL = 0
PER_ROW = 1
PER_CHANNEL = 2
PER_EDGE = 3
ROW_CONST = 4
CHANNEL_CONST = 5
SCALAR = 6

# The autograd wrapper has fixed positional slots; the profiled chains use at
# most four terms over at most eight factors.
_MAX_TERMS = 4
_MAX_FACTORS = 8


def _normalize(factor: torch.Tensor) -> torch.Tensor:
    """Squeeze singleton axes above three dimensions and lift rank 2 to rank 3."""
    if factor.dim() == 0:
        return factor.reshape(1)
    shape = list(factor.shape)
    while len(shape) > 3:
        index = None
        if shape[-2] == 1:
            index = len(shape) - 2
        else:
            for candidate in range(len(shape) - 1):
                if shape[candidate] == 1:
                    index = candidate
                    break
        if index is None:
            raise ValueError(
                f"edge combine factor {tuple(factor.shape)} cannot be reduced to 3D"
            )
        factor = factor.squeeze(index)
        shape = list(factor.shape)
    if len(shape) == 2:
        factor = factor.unsqueeze(2)
    elif len(shape) == 1 and shape[0] != 1:
        raise ValueError(f"edge combine factor {tuple(factor.shape)} is not a scalar")
    return factor


def _shape_of(factors):
    edges = rows = channels = 1
    for factor in factors:
        if factor.dim() <= 1:
            continue
        edges = max(edges, int(factor.shape[0]))
        rows = max(rows, int(factor.shape[1]))
        channels = max(channels, int(factor.shape[2]))
    if edges == 1 and rows == 1 and channels == 1:
        raise ValueError("edge combine needs at least one non-scalar factor")
    return edges, rows, channels


def _mode_of(factor, edges, rows, channels):
    if factor.dim() <= 1:
        if factor.numel() != 1:
            raise ValueError("edge combine 1D factors must be scalars")
        return SCALAR
    shape = tuple(int(value) for value in factor.shape)
    if shape == (edges, rows, channels):
        return FULL
    if shape == (edges, rows, 1):
        return PER_ROW
    if shape == (edges, 1, channels):
        return PER_CHANNEL
    if shape == (edges, 1, 1):
        return PER_EDGE
    if shape == (1, rows, 1):
        return ROW_CONST
    if shape == (1, 1, channels):
        return CHANNEL_CONST
    raise ValueError(
        f"edge combine factor {shape} does not broadcast to ({edges}, {rows}, {channels})"
    )


def _needs_grad(factors):
    return torch.is_grad_enabled() and any(factor.requires_grad for factor in factors)


class _EdgeCombineFunction(torch.autograd.Function):
    """Autograd wrapper: each gradient is coef * the term's other factors."""

    @staticmethod
    def forward(ctx, *args):
        *slots, coefficients, modes, term_starts, term_counts, shape = args
        factors = [factor for factor in slots if factor is not None]
        edges, rows, channels = shape
        out = _extension("edge_combine").edge_combine(
            factors, list(modes), list(coefficients), list(term_starts),
            list(term_counts), edges, rows, channels,
        )
        ctx.save_for_backward(*slots)
        ctx.n_factors = len(factors)
        ctx.coefficients = list(coefficients)
        ctx.modes = list(modes)
        ctx.term_starts = list(term_starts)
        ctx.term_counts = list(term_counts)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        slots = ctx.saved_tensors
        accumulated = [None] * ctx.n_factors
        for term, (start, count) in enumerate(
            zip(ctx.term_starts, ctx.term_counts)
        ):
            coefficient = ctx.coefficients[term]
            indices = list(range(start, start + count))
            for index in indices:
                grad = grad_out if coefficient == 1.0 else grad_out * coefficient
                for other in indices:
                    if other != index:
                        grad = grad * slots[other]
                accumulated[index] = (
                    grad if accumulated[index] is None else accumulated[index] + grad
                )
        grads = [None] * len(slots)
        for index, grad in enumerate(accumulated):
            if grad is None or not ctx.needs_input_grad[index]:
                continue
            mode = ctx.modes[index]
            if mode == PER_ROW:
                grad = grad.sum(dim=2, keepdim=True)
            elif mode == PER_CHANNEL:
                grad = grad.sum(dim=1, keepdim=True)
            elif mode == PER_EDGE:
                grad = grad.sum(dim=(1, 2), keepdim=True)
            elif mode == ROW_CONST:
                grad = grad.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
            elif mode == CHANNEL_CONST:
                grad = grad.sum(dim=0, keepdim=True).sum(dim=1, keepdim=True)
            elif mode == SCALAR:
                grad = grad.sum().reshape(1)
            grads[index] = grad
        return (*grads, None, None, None, None, None)


def edge_combine(*terms, coefficients=None):
    """Return ``sum_t coef_t * prod_f factor_{t,f}`` in one edge-stream pass.

    Each term is a float32 DCU tensor or a tuple of them (a product); at most
    four terms over at most eight factors. Coefficients default to all ones.
    """
    if not terms:
        raise ValueError("edge combine needs at least one term")
    if len(terms) > _MAX_TERMS:
        raise ValueError(f"edge combine supports at most {_MAX_TERMS} terms")
    groups = []
    for term in terms:
        if isinstance(term, torch.Tensor):
            groups.append((term,))
        elif (
            isinstance(term, (tuple, list))
            and term
            and all(isinstance(factor, torch.Tensor) for factor in term)
        ):
            groups.append(tuple(term))
        else:
            raise TypeError("edge combine terms must be tensors or tuples of tensors")
    flat = [factor for group in groups for factor in group]
    if len(flat) > _MAX_FACTORS:
        raise ValueError(f"edge combine supports at most {_MAX_FACTORS} factors")
    for factor in flat:
        if factor.dtype != torch.float32:
            raise TypeError("edge combine factors must be float32")
    coefficients = (
        [1.0] * len(groups)
        if coefficients is None
        else [float(value) for value in coefficients]
    )
    if len(coefficients) != len(groups):
        raise ValueError("edge combine coefficients must match the term count")

    normalized = [_normalize(factor).contiguous() for factor in flat]
    edges, rows, channels = _shape_of(normalized)
    if channels % 4 != 0:
        raise ValueError("edge combine channels must be a multiple of 4")
    modes = [_mode_of(factor, edges, rows, channels) for factor in normalized]
    term_starts, term_counts, cursor = [], [], 0
    for group in groups:
        term_starts.append(cursor)
        term_counts.append(len(group))
        cursor += len(group)

    require_dcu(*normalized)
    padded = normalized + [None] * (_MAX_FACTORS - len(normalized))
    if _needs_grad(normalized):
        return _EdgeCombineFunction.apply(
            *padded, coefficients, modes, term_starts, term_counts,
            (edges, rows, channels),
        )
    return _extension("edge_combine").edge_combine(
        normalized, modes, coefficients, term_starts, term_counts, edges, rows,
        channels,
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
    "ROW_CONST",
    "CHANNEL_CONST",
    "SCALAR",
    "edge_combine",
    "edge_combine_available",
]
