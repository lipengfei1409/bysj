"""Typed/autograd Python interfaces to the DTK PyTorch extension."""
from __future__ import annotations

import importlib

import torch
from torch.autograd.function import once_differentiable


def native_available():
    if not torch.version.hip:
        return False
    try:
        importlib.import_module("dcu_dpa4._C")
    except (ImportError, OSError):
        return False
    return True


def _extension():
    if not torch.version.hip:
        raise RuntimeError("HIP backend requires ROCm/DTK PyTorch")
    try:
        return importlib.import_module("dcu_dpa4._C")
    except (ImportError, OSError) as exc:
        raise RuntimeError("build extension with DPA4_BUILD_HIP=1 python setup.py build_ext --inplace") from exc


class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, edge_index, shifts, cells, edge_frame):
        vec, distance, unit = _extension().edge_geometry(positions, edge_index, shifts, cells, edge_frame)
        ctx.save_for_backward(distance, unit, edge_index, shifts, edge_frame)
        ctx.n, ctx.f = positions.shape[0], cells.shape[0]
        return vec, distance, unit

    @staticmethod
    @once_differentiable
    def backward(ctx, gv, gr, gu):
        r, u, edge, shift, frame = ctx.saved_tensors
        gp, gc = _extension().edge_geometry_backward(gv.contiguous(), gr.contiguous(), gu.contiguous(), r, u, edge, shift, frame, ctx.n, ctx.f)
        return gp, None, None, gc, None


def edge_geometry(positions, edge_index, shifts, cells, edge_frame):
    return _Geometry.apply(positions.contiguous(), edge_index.contiguous(), shifts.contiguous(), cells.contiguous(), edge_frame.contiguous())


class _EdgeFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, edge_index, shifts, cells, edge_frame, mask,
                frequencies, cutoff, basis_exponent, edge_exponent, eps):
        outputs = _extension().edge_features(
            positions, edge_index, shifts, cells, edge_frame, mask, frequencies,
            cutoff, basis_exponent, edge_exponent, eps,
        )
        vec, distance, unit, rbf, env = outputs
        ctx.save_for_backward(vec, distance, unit, edge_index, shifts, cells,
                              edge_frame, mask, frequencies)
        ctx.n_atoms = positions.shape[0]
        ctx.options = cutoff, basis_exponent, edge_exponent, eps
        return vec, distance, unit, rbf, env

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_vec, grad_distance, grad_unit, grad_rbf, grad_env):
        vec, distance, unit, edge_index, shifts, cells, edge_frame, mask, frequencies = ctx.saved_tensors
        grad_vec = torch.zeros_like(vec) if grad_vec is None else grad_vec.contiguous()
        grad_distance = torch.zeros_like(distance) if grad_distance is None else grad_distance.contiguous()
        grad_unit = torch.zeros_like(unit) if grad_unit is None else grad_unit.contiguous()
        grad_rbf = (torch.zeros(vec.shape[0], frequencies.shape[1], dtype=vec.dtype, device=vec.device)
                    if grad_rbf is None else grad_rbf.contiguous())
        grad_env = torch.zeros_like(distance) if grad_env is None else grad_env.contiguous()
        grad_positions, grad_cells, grad_frequencies = _extension().edge_features_backward(
            grad_vec, grad_distance, grad_unit, grad_rbf, grad_env,
            vec, distance, unit, edge_index, shifts, cells, edge_frame, mask,
            frequencies, ctx.n_atoms, *ctx.options,
        )
        return grad_positions, None, None, grad_cells, None, None, grad_frequencies, None, None, None, None


def edge_features(positions, edge_index, shifts, cells, edge_frame, mask,
                  frequencies, cutoff, basis_exponent=7, edge_exponent=5, eps=1e-7):
    return _EdgeFeatures.apply(
        positions.contiguous(), edge_index.contiguous(), shifts.contiguous(),
        cells.contiguous(), edge_frame.contiguous(), mask.contiguous(),
        frequencies.contiguous(), float(cutoff), int(basis_exponent), int(edge_exponent), float(eps),
    )


class _Radial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, cutoff, basis_exponent, edge_exponent):
        ctx.save_for_backward(r, w)
        ctx.options = cutoff, basis_exponent, edge_exponent
        return tuple(_extension().radial(r, w, cutoff, basis_exponent, edge_exponent))

    @staticmethod
    @once_differentiable
    def backward(ctx, g, ge):
        r, w = ctx.saved_tensors
        gr, gw = _extension().radial_backward(g.contiguous(), ge.contiguous(), r, w, *ctx.options)
        return gr, gw, None, None, None


def radial(r, frequencies, rcut, basis_exponent=7, edge_exponent=5):
    return _Radial.apply(r.contiguous(), frequencies.contiguous(), float(rcut), int(basis_exponent), int(edge_exponent))


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, eps):
        ctx.save_for_backward(x, scale)
        ctx.eps = eps
        return _extension().rmsnorm(x, scale, eps)

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        x, scale = ctx.saved_tensors
        gx, gs = _extension().rmsnorm_backward(g.contiguous(), x, scale, ctx.eps)
        return gx, gs, None


def rmsnorm(x, scale, eps=1e-7):
    return _Norm.apply(x.contiguous(), scale.contiguous(), float(eps))


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias.numel() > 0
        return _extension().linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, weight = ctx.saved_tensors
        return g @ weight.T, x.T @ g, g.sum(0) if ctx.has_bias else None


def linear(x, weight, bias=None):
    if bias is None:
        bias = x.new_empty(0)
    return _Linear.apply(x.contiguous(), weight.contiguous(), bias.contiguous())


class _Segment(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, index, size):
        ctx.save_for_backward(index)
        return _extension().segment_sum(x, index, size)

    @staticmethod
    def backward(ctx, g):
        (index,) = ctx.saved_tensors
        return g.index_select(0, index), None, None


def segment_sum(x, index, size):
    return _Segment.apply(x.contiguous(), index.contiguous(), int(size))


def force_virial(gradient, vectors, edge_index, edge_frame, mask, n_atoms, n_frames):
    if gradient.requires_grad or vectors.requires_grad:
        raise NotImplementedError("native force/virial assembly currently supports inference, not force-loss backward")
    return tuple(_extension().force_virial(gradient.contiguous(), vectors.contiguous(), edge_index.contiguous(),
        edge_frame.contiguous(), mask.contiguous(), int(n_atoms), int(n_frames)))
