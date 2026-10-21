"""Uniform primitive interfaces: backend='reference' or backend='hip'."""
from __future__ import annotations

import math
import torch

from .radial import radial_basis, c3_cutoff_envelope


def _native(backend):
    if backend not in ("reference", "hip"):
        raise ValueError("backend must be reference or hip")
    if backend == "hip":
        from . import native
        return native
    return None


class _FusedEdgeScalarFeatures(torch.autograd.Function):
    """Keep fused scalar feature values differentiable through the edge vector."""

    @staticmethod
    def forward(ctx, vec, distance, unit, rbf, env, frequencies, mask,
                cutoff, basis_exponent, edge_exponent, eps):
        ctx.save_for_backward(vec, distance, unit, frequencies, mask)
        ctx.options = float(cutoff), int(basis_exponent), int(edge_exponent), float(eps)
        return distance, unit, rbf, env

    @staticmethod
    def backward(ctx, grad_distance, grad_unit, grad_rbf, grad_env):
        vec, distance, unit, frequencies, mask = ctx.saved_tensors
        cutoff, basis_exponent, edge_exponent, _ = ctx.options
        grad_vec = torch.zeros_like(vec)
        grad_frequency = torch.zeros_like(frequencies)
        if vec.numel() == 0:
            return grad_vec, None, None, None, None, grad_frequency, None, None, None, None, None

        r = distance
        active = mask[:, None]
        active_float = active.to(vec.dtype)
        grad_r = torch.zeros_like(r)
        if grad_distance is not None:
            grad_r = grad_r + grad_distance * active_float
        if grad_unit is not None:
            grad_unit = grad_unit * active_float
            unit_dot = (grad_unit * unit).sum(dim=-1, keepdim=True)
            grad_vec = grad_vec + (grad_unit - unit * unit_dot) / r

        if grad_env is not None and edge_exponent:
            _, env_derivative = _c3_envelope_pair(r, cutoff, edge_exponent)
            grad_r = grad_r + grad_env * active_float * env_derivative
        if grad_rbf is not None:
            grad_rbf = grad_rbf * active_float
            basis_env, basis_derivative = _c3_envelope_pair(r, cutoff, basis_exponent)
            z = r * frequencies
            sinc, sinc_derivative = _sinc_pair(z)
            grad_r = grad_r + (
                grad_rbf * (frequencies.square() * sinc_derivative * basis_env
                            + frequencies * sinc * basis_derivative)
            ).sum(dim=-1, keepdim=True)
            grad_frequency = (
                grad_rbf * (sinc + z * sinc_derivative) * basis_env
            ).sum(dim=0, keepdim=True)

        grad_vec = grad_vec + (grad_r / r) * vec
        grad_vec = torch.where(active, grad_vec, 0.)
        return grad_vec, None, None, None, None, grad_frequency, None, None, None, None, None


def _c3_envelope_pair(distance, cutoff, exponent):
    if exponent == 0:
        return torch.ones_like(distance), torch.zeros_like(distance)
    x = (distance / cutoff).clamp(min=0.0, max=1.0)
    u = 1.0 - x
    series = torch.zeros_like(x)
    derivative_series = torch.zeros_like(x)
    for k in range(exponent):
        coefficient = math.comb(k + 3, 3)
        series = series + coefficient * x.pow(k)
        if k:
            derivative_series = derivative_series + k * coefficient * x.pow(k - 1)
    value = u.pow(4) * series
    derivative = (-4.0 * u.pow(3) * series + u.pow(4) * derivative_series) / cutoff
    inside = distance < cutoff
    return torch.where(inside, value, 0.0), torch.where(inside, derivative, 0.0)


def _sinc_pair(z):
    small = z.abs() < 1.0e-3
    safe = torch.where(small, torch.ones_like(z), z)
    sinc_regular = torch.sin(z) / safe
    derivative_regular = (z * torch.cos(z) - torch.sin(z)) / safe.square()
    z2 = z.square()
    sinc_small = 1.0 - z2 / 6.0 + z2.square() / 120.0
    derivative_small = -z / 3.0 + z * z2 / 30.0
    return torch.where(small, sinc_small, sinc_regular), torch.where(small, derivative_small, derivative_regular)


def edge_geometry(positions, edge_index, shifts, cells, edge_frame, *, backend="reference"):
    native = _native(backend)
    if native:
        return native.edge_geometry(positions, edge_index, shifts, cells, edge_frame)
    if positions.ndim != 2 or positions.shape[1] != 3 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("positions must be [N,3] and edge_index [2,E]")
    e = edge_index.shape[1]
    if shifts.shape != (e, 3) or edge_frame.shape != (e,) or cells.ndim != 3 or cells.shape[1:] != (3, 3):
        raise ValueError("shifts, edge_frame or cells have invalid shape")
    src, dst = edge_index
    displacement = positions[src] - positions[dst] + torch.einsum("ei,eij->ej", shifts.to(positions.dtype), cells[edge_frame])
    distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
    safe = torch.where(distance > 0, distance, torch.ones_like(distance))
    unit = torch.where(distance > 0, displacement / safe, torch.zeros_like(displacement))
    return displacement, distance, unit


def edge_features(positions, edge_index, shifts, cells, edge_frame, mask,
                  frequencies, rcut, basis_exponent=7, edge_exponent=5,
                  eps=1e-7, *, backend="reference"):
    """Fused edge geometry/radial interface; HIP VJP covers every output."""
    native = _native(backend)
    if eps <= 0:
        raise ValueError("geometry epsilon must be positive")
    if native:
        vec, distance, unit, rbf, env = native.edge_features(
            positions, edge_index, shifts, cells, edge_frame, mask,
            frequencies, rcut, basis_exponent, edge_exponent, eps,
        )
        distance, unit, rbf, env = _FusedEdgeScalarFeatures.apply(
            vec, distance, unit, rbf, env, frequencies, mask,
            rcut, basis_exponent, edge_exponent, eps,
        )
        return vec, distance, unit, rbf, env
    vec, _, _ = edge_geometry(positions, edge_index, shifts, cells, edge_frame, backend="reference")
    vec = torch.where(mask[:, None], vec, vec.new_tensor([0., 0., 1.]))
    distance = torch.sqrt(vec.square().sum(-1, keepdim=True) + float(eps) ** 2)
    unit = vec / distance
    rbf, env = radial(distance, frequencies, rcut, basis_exponent, edge_exponent, backend="reference")
    rbf = torch.where(mask[:, None], rbf, 0.)
    env = torch.where(mask[:, None], env, 0.)
    return vec, distance, unit, rbf, env


def edge_features_persistent(positions, edge_index, shifts, cells, edge_frame, mask,
                             frequencies, rcut, basis_exponent=7, edge_exponent=5,
                             eps=1e-7, *, runtime_plan):
    """Execute a selected edge-feature task plan with persistent HIP workers."""
    if eps <= 0:
        raise ValueError("geometry epsilon must be positive")
    from . import native
    vec, distance, unit, rbf, env, status = native.edge_features_persistent(
        positions, edge_index, shifts, cells, edge_frame, mask, frequencies,
        rcut, basis_exponent, edge_exponent, eps, runtime_plan=runtime_plan,
    )
    distance, unit, rbf, env = _FusedEdgeScalarFeatures.apply(
        vec, distance, unit, rbf, env, frequencies, mask,
        rcut, basis_exponent, edge_exponent, eps,
    )
    return vec, distance, unit, rbf, env, status


def radial(r, frequencies, rcut, basis_exponent=7, edge_exponent=5, *, backend="reference"):
    native = _native(backend)
    if native:
        return native.radial(r, frequencies, rcut, basis_exponent, edge_exponent)
    env = c3_cutoff_envelope(r, rcut, edge_exponent) if edge_exponent else torch.ones_like(r)
    return radial_basis(r, frequencies, rcut=rcut, exponent=basis_exponent), env


def rmsnorm(x, scale, eps=1e-7, *, backend="reference"):
    native = _native(backend)
    if native:
        return native.rmsnorm(x, scale, eps)
    if x.ndim != 2 or scale.shape != (x.shape[1],) or x.shape[1] == 0 or eps <= 0:
        raise ValueError("RMSNorm requires x [N,C], scale [C] and eps>0")
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * scale


def linear(x, weight, bias=None, *, backend="reference"):
    native = _native(backend)
    if native:
        return native.linear(x, weight, bias)
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError("linear requires x [N,K], weight [K,C]")
    result = x @ weight
    return result if bias is None else result + bias


def segment_sum(x, index, size, *, backend="reference"):
    native = _native(backend)
    if native:
        return native.segment_sum(x, index, size)
    if x.ndim != 2 or index.shape != (x.shape[0],) or size < 0:
        raise ValueError("segment_sum requires x [E,K], index [E], size>=0")
    return x.new_zeros((size, x.shape[1])).index_add(0, index, x)


def force_virial(gradient, vectors, edge_index, edge_frame, mask, n_atoms, n_frames, *, backend="reference"):
    native = _native(backend)
    if native:
        return native.force_virial(gradient, vectors, edge_index, edge_frame, mask, n_atoms, n_frames)
    if gradient.shape != vectors.shape or gradient.ndim != 2 or gradient.shape[1] != 3:
        raise ValueError("gradient/vectors must be [E,3]")
    e = gradient.shape[0]
    if edge_index.shape != (2, e) or mask.shape != (e,) or edge_frame.shape != (e,) or n_atoms < 0 or n_frames < 1:
        raise ValueError("force/virial topology shapes disagree")
    g = torch.where(mask[:, None], gradient, 0.0)
    force = gradient.new_zeros((n_atoms, 3))
    force = force.index_add(0, edge_index[1], g).index_add(0, edge_index[0], -g)
    edge_virial = -torch.einsum("ek,ej->ekj", g, vectors).reshape(e, 9)
    virial = segment_sum(edge_virial, edge_frame, n_frames).reshape(n_frames, 3, 3)
    return force, virial
