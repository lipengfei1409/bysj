"""DPA4/SeZM radial reference, matching the checkpoint's Bessel convention."""

from __future__ import annotations

import math

import torch


def c3_cutoff_envelope(
    distances: torch.Tensor, rcut: float, exponent: int
) -> torch.Tensor:
    """C³ cutoff: ``(1-x)^4 Σ C(k+3,3)x^k`` for ``x=r/rcut < 1``."""
    if rcut <= 0 or exponent <= 0:
        raise ValueError("rcut and exponent must be positive")
    if distances.ndim != 2 or distances.shape[1] != 1:
        raise ValueError("distances must have shape [E, 1]")
    u = ((rcut - distances) / rcut).clamp(min=0.0, max=1.0)
    x = 1.0 - u
    coefficients = [math.comb(k + 3, 3) for k in range(exponent)]
    series = torch.full_like(x, float(coefficients[-1]))
    for coefficient in reversed(coefficients[:-1]):
        series = coefficient + x * series
    return u.pow(4) * series


def radial_basis(
    distances: torch.Tensor,
    frequencies: torch.Tensor,
    *,
    rcut: float,
    exponent: int = 7,
    basis_type: str = "bessel",
) -> torch.Tensor:
    """Evaluate learned radial frequencies; output shape ``[E, R]``."""
    if distances.ndim != 2 or distances.shape[1] != 1:
        raise ValueError("distances must have shape [E, 1]")
    if frequencies.ndim != 2 or frequencies.shape[0] != 1:
        raise ValueError("frequencies must have shape [1, R]")
    if distances.device != frequencies.device:
        raise ValueError("distances and frequencies must share a device")
    if rcut <= 0 or exponent < 0:
        raise ValueError("rcut must be positive and exponent nonnegative")
    family = basis_type.removesuffix("/fix")
    if family == "bessel":
        raw = frequencies * torch.sinc((distances * frequencies) / math.pi)
    elif family == "gaussian":
        width = rcut / max(frequencies.shape[1] - 1, 1)
        raw = torch.exp(-0.5 * ((distances - frequencies) / width).square())
    else:
        raise ValueError(f"unsupported basis_type: {basis_type}")
    if exponent:
        return raw * c3_cutoff_envelope(distances, rcut, exponent)
    return raw
