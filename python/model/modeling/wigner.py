# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reference Wigner-D rotation for the packed SeZM real basis."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class _CaseTerms:
    coefficient: float
    ra_power: int
    rb_power: int
    sign: float
    horner: tuple[float, ...]

    @property
    def valid(self):
        return self.coefficient != 0.0


def _make_case(ell: int, mp: int, m: int, *, case1: bool) -> _CaseTerms:
    """Build one stable quaternion-polynomial branch for a Wigner entry."""
    if case1:
        rho_min = max(0, mp - m)
        rho_max = min(ell + mp, ell - m)
        if rho_min > rho_max:
            return _CaseTerms(0.0, 0, 0, 0.0, ())
        binom1 = math.comb(ell + mp, rho_min)
        binom2 = math.comb(ell - mp, ell - m - rho_min)
    else:
        rho_min = max(0, -(mp + m))
        rho_max = min(ell - m, ell - mp)
        if rho_min > rho_max:
            return _CaseTerms(0.0, 0, 0, 0.0, ())
        binom1 = math.comb(ell + mp, ell - m - rho_min)
        binom2 = math.comb(ell - mp, rho_min)

    sqrt_factor = math.sqrt(
        math.factorial(ell + m) * math.factorial(ell - m)
        / (math.factorial(ell + mp) * math.factorial(ell - mp))
    )
    horner = []
    for rho in range(rho_max, rho_min, -1):
        if case1:
            n1, n2 = ell + mp - rho + 1, ell - m - rho + 1
            d1, d2 = rho, m - mp + rho
        else:
            n1, n2 = ell - m - rho + 1, ell - mp - rho + 1
            d1, d2 = rho, mp + m + rho
        horner.append((n1 * n2) / (d1 * d2) if d1 and d2 else 0.0)

    if case1:
        ra_power = 2 * ell + mp - m - 2 * rho_min
        rb_power = m - mp + 2 * rho_min
        sign = (-1.0) ** rho_min
    else:
        ra_power = mp + m + 2 * rho_min
        rb_power = 2 * ell - mp - m - 2 * rho_min
        sign = ((-1.0) ** (ell - m)) * ((-1.0) ** rho_min)
    return _CaseTerms(sqrt_factor * binom1 * binom2, ra_power, rb_power, sign, tuple(horner))


def _normalize_quaternion(q: torch.Tensor, eps: float) -> torch.Tensor:
    return q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + eps * eps)


def _smooth_step(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    eps = torch.finfo(x.dtype).eps
    left = torch.exp(-1.0 / torch.clamp(x, min=eps))
    right = torch.exp(-1.0 / torch.clamp(1.0 - x, min=eps))
    interior = left / (left + right)
    return torch.where(x <= 0.0, torch.zeros_like(x),
                       torch.where(x >= 1.0, torch.ones_like(x), interior))


def _edge_quaternion(edge_unit: torch.Tensor, eps: float) -> torch.Tensor:
    """Smoothly blend the two exact quaternion charts that align an edge to +Z."""
    x, y, z = edge_unit.unbind(dim=-1)
    q_pos = _normalize_quaternion(torch.stack((1.0 + z, y, -x, torch.zeros_like(x)), dim=-1), eps)
    q_neg = _normalize_quaternion(torch.stack((-x, torch.zeros_like(x), 1.0 - z, y), dim=-1), eps)
    blend = _smooth_step(0.5 * (z + 1.0))
    dot = (q_neg * q_pos).sum(dim=-1, keepdim=True)
    q_pos = torch.where(dot < 0.0, -q_pos, q_pos)
    return _normalize_quaternion((1.0 - blend[:, None]) * q_neg + blend[:, None] * q_pos, eps)


def _quaternion_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), dim=-1),
        torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), dim=-1),
        torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1),
    ), dim=-2)


def _complex_to_real_basis(ell: int, device: torch.device) -> torch.Tensor:
    size = 2 * ell + 1
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    real = torch.zeros(size, size, dtype=torch.float64, device=device)
    imag = torch.zeros_like(real)
    for m in range(-ell, ell + 1):
        row = m + ell
        if m == 0:
            real[row, ell] = 1.0
        elif m > 0:
            real[row, m + ell] = inv_sqrt2
            real[row, -m + ell] = ((-1) ** m) * inv_sqrt2
        else:
            imag[row, -m + ell] = -inv_sqrt2
            imag[row, m + ell] = ((-1) ** m) * inv_sqrt2
    return torch.complex(real, imag)


def _horner_magnitude(terms: _CaseTerms, ratio: torch.Tensor,
                      log_ra: torch.Tensor, log_rb: torch.Tensor) -> torch.Tensor:
    value = torch.ones_like(ratio)
    for coefficient in terms.horner:
        value = 1.0 + value * (ratio * coefficient)
    return (terms.sign * terms.coefficient
            * torch.exp(log_ra * terms.ra_power + log_rb * terms.rb_power)
            * value)


def _complex_wigner_block(q: torch.Tensor, ell: int) -> torch.Tensor:
    """Evaluate one Wigner block using the generic SeZM quaternion formula."""
    q = q.to(torch.float64)
    w, x, y, z = q.unbind(dim=-1)
    ra_re, ra_im, rb_re, rb_im = w, -z, y, -x
    ra_sq = ra_re.square() + ra_im.square()
    rb_sq = rb_re.square() + rb_im.square()
    eps = torch.finfo(torch.float64).eps
    eps_sq = eps * eps
    ra_small, rb_small = ra_sq <= eps_sq, rb_sq <= eps_sq
    ra, rb = torch.sqrt(ra_sq.clamp_min(eps_sq)), torch.sqrt(rb_sq.clamp_min(eps_sq))
    safe_ra_re = torch.where(ra_small, torch.ones_like(ra_re), ra_re)
    safe_ra_im = torch.where(ra_small, torch.zeros_like(ra_im), ra_im)
    safe_rb_re = torch.where(rb_small, torch.ones_like(rb_re), rb_re)
    safe_rb_im = torch.where(rb_small, torch.zeros_like(rb_im), rb_im)
    phi_a = torch.atan2(safe_ra_im, safe_ra_re)
    phi_b = torch.atan2(safe_rb_im, safe_rb_re)
    safe_ra, safe_rb = ra.clamp_min(eps), rb.clamp_min(eps)
    log_ra, log_rb = torch.log(safe_ra), torch.log(safe_rb)
    ratio1 = -(rb.square()) / safe_ra.square()
    ratio2 = -(ra.square()) / safe_rb.square()
    use_case1 = (ra >= rb) & ~ra_small & ~rb_small
    use_case2 = (ra < rb) & ~ra_small & ~rb_small
    size = 2 * ell + 1
    real = torch.zeros(q.shape[0], size, size, dtype=torch.float64, device=q.device)
    imag = torch.zeros_like(real)

    for mp in range(-ell, ell + 1):
        for m in range(-ell, ell + 1):
            primary = (mp + m > 0) or (mp + m == 0 and mp >= 0)
            if not primary:
                continue
            case1 = _make_case(ell, mp, m, case1=True)
            case2 = _make_case(ell, mp, m, case1=False)
            magnitude1 = _horner_magnitude(case1, ratio1, log_ra, log_rb)
            magnitude2 = _horner_magnitude(case2, ratio2, log_ra, log_rb)
            magnitude = torch.where(use_case1, magnitude1,
                                    torch.where(use_case2, magnitude2, torch.zeros_like(magnitude1)))
            phase = (mp + m) * phi_a + (m - mp) * phi_b
            entry_re = magnitude * torch.cos(phase)
            entry_im = magnitude * torch.sin(phase)
            if mp == -m:
                anti = ra_small
                anti_mag = ((-1.0) ** (ell - m)) * torch.exp(2 * m * log_rb)
                anti_phase = 2 * m * phi_b
                entry_re = torch.where(anti, anti_mag * torch.cos(anti_phase), entry_re)
                entry_im = torch.where(anti, anti_mag * torch.sin(anti_phase), entry_im)
            if mp == m:
                diagonal = rb_small
                diagonal_mag = torch.exp(2 * m * log_ra)
                diagonal_phase = 2 * m * phi_a
                entry_re = torch.where(diagonal, diagonal_mag * torch.cos(diagonal_phase), entry_re)
                entry_im = torch.where(diagonal, diagonal_mag * torch.sin(diagonal_phase), entry_im)
            row, col = mp + ell, m + ell
            real[:, row, col] = entry_re
            imag[:, row, col] = entry_im

    for mp in range(-ell, ell + 1):
        for m in range(-ell, ell + 1):
            if (mp + m > 0) or (mp + m == 0 and mp >= 0):
                continue
            row, col = mp + ell, m + ell
            primary_row, primary_col = -mp + ell, -m + ell
            sign = (-1.0) ** (mp - m)
            real[:, row, col] = sign * real[:, primary_row, primary_col]
            imag[:, row, col] = -sign * imag[:, primary_row, primary_col]

    d_complex = torch.complex(real, imag)
    basis = _complex_to_real_basis(ell, q.device)
    return (basis @ d_complex @ basis.conj().transpose(-1, -2)).real


class WignerRotation(nn.Module):
    """Reference Wigner rotation with the checkpoint's l=1 basis convention."""

    def __init__(self, lmax: int, l1_perm: torch.Tensor, l1_sign_outer: torch.Tensor, *, eps=1e-7):
        super().__init__()
        if lmax < 0 or lmax > 10:
            raise ValueError('reference Wigner rotation supports 0 <= lmax <= 10')
        if l1_perm.shape != (3,) or l1_sign_outer.shape != (3, 3):
            raise ValueError('checkpoint l=1 Wigner basis buffers have invalid shapes')
        self.lmax, self.eps = int(lmax), float(eps)
        self.register_buffer('l1_perm', l1_perm.detach().clone())
        self.register_buffer('l1_sign_outer', l1_sign_outer.detach().clone())

    def forward(self, unit):
        if unit.ndim != 2 or unit.shape[1] != 3:
            raise ValueError('Wigner input unit vectors must have shape [E,3]')
        q = _edge_quaternion(unit, self.eps)
        dim = (self.lmax + 1) ** 2
        d_full = unit.new_zeros((unit.shape[0], dim, dim))
        d_full[:, 0, 0] = 1.0
        if self.lmax >= 1:
            rot = _quaternion_rotation_matrix(q.to(unit.dtype))
            block1 = rot.index_select(-2, self.l1_perm).index_select(-1, self.l1_perm)
            d_full[:, 1:4, 1:4] = block1 * self.l1_sign_outer
        for ell in range(2, self.lmax + 1):
            start = ell * ell
            stop = (ell + 1) * (ell + 1)
            d_full[:, start:stop, start:stop] = _complex_wigner_block(q, ell).to(unit.dtype)
        return {'D': d_full, 'Dt': d_full.transpose(-1, -2)}
