"""Standalone TileLang edge-aligned quaternion construction for DPA4."""

from functools import lru_cache

import torch

_BLOCK = 128
_FLOAT32_EPS = 1.1920928955078125e-7


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang edge quaternion requires TileLang to be installed"
        ) from error
    return tilelang, T


def _chart_values(x, y, z, chart_length, eps, T):
    """Return the two normalized quaternion charts for one edge."""
    ux = x / chart_length
    uy = y / chart_length
    uz = z / chart_length

    p0 = 1.0 + uz
    p1 = uy
    p2 = -ux
    pnorm = T.sqrt(p0 * p0 + p1 * p1 + p2 * p2 + eps * eps)
    qp = (p0 / pnorm, p1 / pnorm, p2 / pnorm, 0.0)

    n0 = -ux
    n2 = 1.0 - uz
    n3 = uy
    nnorm = T.sqrt(n0 * n0 + n2 * n2 + n3 * n3 + eps * eps)
    qn = (n0 / nnorm, 0.0, n2 / nnorm, n3 / nnorm)
    return ux, uy, uz, qp, qn


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def edge_quaternion_forward(n_edges, block, eps, has_edge_len, dtype=T.float32):
        @T.prim_func
        def main(
            edge_vec: T.Tensor((n_edges, 3), dtype),
            edge_len: T.Tensor((n_edges, 1), dtype),
            edge_quat: T.Tensor((n_edges, 4), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block), threads=128) as (pid,):
                for edge_offset in T.Parallel(block):
                    edge = pid * block + edge_offset
                    if edge < n_edges:
                        x = edge_vec[edge, 0]
                        y = edge_vec[edge, 1]
                        z = edge_vec[edge, 2]
                        vector_length = T.sqrt(x * x + y * y + z * z + eps * eps)
                        input_length = T.sqrt(
                            edge_len[edge, 0] * edge_len[edge, 0] + eps * eps
                        )
                        if has_edge_len:
                            chart_length = input_length
                        else:
                            chart_length = vector_length
                        _, _, uz, qp, qn = _chart_values(x, y, z, chart_length, eps, T)
                        blend_x = 0.5 * (uz + 1.0)
                        clamped_x = T.min(T.max(blend_x, 0.0), 1.0)
                        left_x = T.max(clamped_x, _FLOAT32_EPS)
                        right_x = T.max(1.0 - clamped_x, _FLOAT32_EPS)
                        left = T.exp(-1.0 / left_x)
                        right = T.exp(-1.0 / right_x)
                        weight = T.if_then_else(
                            clamped_x <= 0.0,
                            0.0,
                            T.if_then_else(
                                clamped_x >= 1.0, 1.0, left / (left + right)
                            ),
                        )
                        dot = (
                            qn[0] * qp[0]
                            + qn[1] * qp[1]
                            + qn[2] * qp[2]
                            + qn[3] * qp[3]
                        )
                        sign = T.if_then_else(dot < 0.0, -1.0, 1.0)
                        b0 = (1.0 - weight) * qn[0] + weight * sign * qp[0]
                        b1 = (1.0 - weight) * qn[1] + weight * sign * qp[1]
                        b2 = (1.0 - weight) * qn[2] + weight * sign * qp[2]
                        b3 = (1.0 - weight) * qn[3] + weight * sign * qp[3]
                        norm = T.sqrt(b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3 + eps * eps)
                        edge_quat[edge, 0] = b0 / norm
                        edge_quat[edge, 1] = b1 / norm
                        edge_quat[edge, 2] = b2 / norm
                        edge_quat[edge, 3] = b3 / norm

        return main

    return edge_quaternion_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def edge_quaternion_backward(n_edges, block, eps, has_edge_len, dtype=T.float32):
        @T.prim_func
        def main(
            edge_vec: T.Tensor((n_edges, 3), dtype),
            edge_len: T.Tensor((n_edges, 1), dtype),
            grad_quat: T.Tensor((n_edges, 4), dtype),
            grad_edge_vec: T.Tensor((n_edges, 3), dtype),
            grad_edge_len: T.Tensor((n_edges, 1), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block), threads=128) as (pid,):
                for edge_offset in T.Parallel(block):
                    edge = pid * block + edge_offset
                    if edge < n_edges:
                        x = edge_vec[edge, 0]
                        y = edge_vec[edge, 1]
                        z = edge_vec[edge, 2]
                        vector_length = T.sqrt(x * x + y * y + z * z + eps * eps)
                        input_length = T.sqrt(
                            edge_len[edge, 0] * edge_len[edge, 0] + eps * eps
                        )
                        if has_edge_len:
                            chart_length = input_length
                        else:
                            chart_length = vector_length
                        ux, uy, uz, qp, qn = _chart_values(
                            x, y, z, chart_length, eps, T
                        )
                        blend_x = 0.5 * (uz + 1.0)
                        clamped_x = T.min(T.max(blend_x, 0.0), 1.0)
                        left_x = T.max(clamped_x, _FLOAT32_EPS)
                        right_x = T.max(1.0 - clamped_x, _FLOAT32_EPS)
                        left = T.exp(-1.0 / left_x)
                        right = T.exp(-1.0 / right_x)
                        weight = T.if_then_else(
                            clamped_x <= 0.0,
                            0.0,
                            T.if_then_else(
                                clamped_x >= 1.0, 1.0, left / (left + right)
                            ),
                        )
                        dot_charts = (
                            qn[0] * qp[0]
                            + qn[1] * qp[1]
                            + qn[2] * qp[2]
                            + qn[3] * qp[3]
                        )
                        sign = T.if_then_else(dot_charts < 0.0, -1.0, 1.0)
                        a0 = sign * qp[0]
                        a1 = sign * qp[1]
                        a2 = sign * qp[2]
                        a3 = sign * qp[3]
                        b0 = (1.0 - weight) * qn[0] + weight * a0
                        b1 = (1.0 - weight) * qn[1] + weight * a1
                        b2 = (1.0 - weight) * qn[2] + weight * a2
                        b3 = (1.0 - weight) * qn[3] + weight * a3
                        norm = T.sqrt(b0 * b0 + b1 * b1 + b2 * b2 + b3 * b3 + eps * eps)
                        q0 = b0 / norm
                        q1 = b1 / norm
                        q2 = b2 / norm
                        q3 = b3 / norm
                        g0 = grad_quat[edge, 0]
                        g1 = grad_quat[edge, 1]
                        g2 = grad_quat[edge, 2]
                        g3 = grad_quat[edge, 3]
                        qdotg = q0 * g0 + q1 * g1 + q2 * g2 + q3 * g3
                        gb0 = (g0 - q0 * qdotg) / norm
                        gb1 = (g1 - q1 * qdotg) / norm
                        gb2 = (g2 - q2 * qdotg) / norm
                        gb3 = (g3 - q3 * qdotg) / norm

                        gw = (
                            gb0 * (a0 - qn[0])
                            + gb1 * (a1 - qn[1])
                            + gb2 * (a2 - qn[2])
                            + gb3 * (a3 - qn[3])
                        )
                        gqn0 = (1.0 - weight) * gb0
                        gqn1 = (1.0 - weight) * gb1
                        gqn2 = (1.0 - weight) * gb2
                        gqn3 = (1.0 - weight) * gb3
                        gqp0 = weight * sign * gb0
                        gqp1 = weight * sign * gb1
                        gqp2 = weight * sign * gb2
                        gqp3 = weight * sign * gb3

                        pnorm = T.sqrt(
                            (1.0 + uz) * (1.0 + uz) + uy * uy + ux * ux + eps * eps
                        )
                        pdot = qp[0] * gqp0 + qp[1] * gqp1 + qp[2] * gqp2 + qp[3] * gqp3
                        gp0 = (gqp0 - qp[0] * pdot) / pnorm
                        gp1 = (gqp1 - qp[1] * pdot) / pnorm
                        gp2 = (gqp2 - qp[2] * pdot) / pnorm

                        nnorm = T.sqrt(
                            ux * ux + (1.0 - uz) * (1.0 - uz) + uy * uy + eps * eps
                        )
                        ndot = qn[0] * gqn0 + qn[1] * gqn1 + qn[2] * gqn2 + qn[3] * gqn3
                        gn0 = (gqn0 - qn[0] * ndot) / nnorm
                        gn2 = (gqn2 - qn[2] * ndot) / nnorm
                        gn3 = (gqn3 - qn[3] * ndot) / nnorm

                        gux = -gp2 - gn0
                        guy = gp1 + gn3
                        guz = gp0 - gn2
                        smooth_active = (clamped_x > _FLOAT32_EPS) & (
                            clamped_x < 1.0 - _FLOAT32_EPS
                        )
                        dw_dx = (
                            weight
                            * (1.0 - weight)
                            * (
                                1.0 / (clamped_x * clamped_x)
                                + 1.0 / ((1.0 - clamped_x) * (1.0 - clamped_x))
                            )
                        )
                        guz += T.if_then_else(smooth_active, 0.5 * gw * dw_dx, 0.0)

                        dot_uv = gux * x + guy * y + guz * z
                        inv = 1.0 / chart_length
                        inv3 = inv * inv * inv
                        grad_x = gux * inv
                        grad_y = guy * inv
                        grad_z = guz * inv
                        if has_edge_len:
                            grad_edge_len[edge, 0] = -dot_uv * edge_len[edge, 0] * inv3
                        else:
                            grad_x -= x * dot_uv * inv3
                            grad_y -= y * dot_uv * inv3
                            grad_z -= z * dot_uv * inv3
                            grad_edge_len[edge, 0] = 0.0
                        grad_edge_vec[edge, 0] = grad_x
                        grad_edge_vec[edge, 1] = grad_y
                        grad_edge_vec[edge, 2] = grad_z

        return main

    return edge_quaternion_backward


def _check_inputs(edge_vec, edge_len):
    if edge_vec.ndim != 2 or edge_vec.shape[1] != 3:
        raise ValueError("edge_vec must have shape (n_edges, 3)")
    if edge_vec.dtype != torch.float32:
        raise TypeError("TileLang edge quaternion requires float32 input")
    if edge_len is not None:
        if edge_len.shape != (edge_vec.shape[0], 1):
            raise ValueError("edge_len must have shape (n_edges, 1)")
        if edge_len.dtype != edge_vec.dtype or edge_len.device != edge_vec.device:
            raise ValueError("edge_len must match edge_vec dtype and device")
    if not edge_vec.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang edge quaternion requires a HIP/DCU tensor")


def _edge_length_input(edge_vec, edge_len):
    if edge_len is not None:
        return edge_len.contiguous()
    return torch.zeros(
        (edge_vec.shape[0], 1), dtype=edge_vec.dtype, device=edge_vec.device
    )


def _forward(edge_vec, edge_len, eps, has_edge_len):
    n_edges = edge_vec.shape[0]
    edge_quat = torch.empty((n_edges, 4), dtype=edge_vec.dtype, device=edge_vec.device)
    if n_edges:
        _, T = _tilelang_modules()
        _forward_factory()(n_edges, _BLOCK, eps, has_edge_len, T.float32)(
            edge_vec.contiguous(), edge_len.contiguous(), edge_quat
        )
    return edge_quat


def _backward(grad_quat, edge_vec, edge_len, eps, has_edge_len):
    grad_edge_vec = torch.empty_like(edge_vec)
    grad_edge_len = torch.empty_like(edge_len)
    if edge_vec.shape[0]:
        _, T = _tilelang_modules()
        _backward_factory()(edge_vec.shape[0], _BLOCK, eps, has_edge_len, T.float32)(
            edge_vec.contiguous(),
            edge_len.contiguous(),
            grad_quat.contiguous(),
            grad_edge_vec,
            grad_edge_len,
        )
    return grad_edge_vec, grad_edge_len


class TileLangEdgeQuaternion(torch.autograd.Function):
    """Construct the smooth two-chart edge quaternion with analytic backward."""

    @staticmethod
    def forward(ctx, edge_vec, edge_len=None, eps=1.0e-7):
        _check_inputs(edge_vec, edge_len)
        edge_vec = edge_vec.contiguous()
        ctx.has_edge_len = edge_len is not None
        edge_len = _edge_length_input(edge_vec, edge_len)
        ctx.eps = float(eps)
        ctx.save_for_backward(edge_vec, edge_len)
        return _forward(edge_vec, edge_len, ctx.eps, ctx.has_edge_len)

    @staticmethod
    def backward(ctx, grad_quat):
        edge_vec, edge_len = ctx.saved_tensors
        if not ctx.needs_input_grad[0] and not (
            ctx.has_edge_len and ctx.needs_input_grad[1]
        ):
            return None, None, None
        grad_edge_vec, grad_edge_len = _backward(
            grad_quat, edge_vec, edge_len, ctx.eps, ctx.has_edge_len
        )
        return (
            grad_edge_vec if ctx.needs_input_grad[0] else None,
            grad_edge_len if ctx.has_edge_len and ctx.needs_input_grad[1] else None,
            None,
        )


def tilelang_edge_quaternion(edge_vec, eps=1.0e-7, *, edge_len=None):
    """Build DPA4 edge quaternions, optionally using a cached edge length."""
    return TileLangEdgeQuaternion.apply(edge_vec, edge_len, float(eps))


def edge_quaternion_reference(edge_vec, edge_len=None, eps=1.0e-7):
    """Differentiable PyTorch reference matching ``build_edge_quaternion``."""
    if edge_len is None:
        edge_len = torch.sqrt((edge_vec * edge_vec).sum(-1, keepdim=True) + eps * eps)
    else:
        edge_len = torch.sqrt(edge_len * edge_len + eps * eps)
    unit = edge_vec / edge_len
    x, y, z = unit.unbind(-1)
    q_pos_raw = torch.stack((1.0 + z, y, -x, torch.zeros_like(x)), dim=-1)
    q_neg_raw = torch.stack((-x, torch.zeros_like(x), 1.0 - z, y), dim=-1)
    q_pos = q_pos_raw / torch.sqrt(
        (q_pos_raw * q_pos_raw).sum(-1, keepdim=True) + eps * eps
    )
    q_neg = q_neg_raw / torch.sqrt(
        (q_neg_raw * q_neg_raw).sum(-1, keepdim=True) + eps * eps
    )
    blend_x = 0.5 * (z + 1.0)
    x_clamped = blend_x.clamp(0.0, 1.0)
    machine_eps = torch.finfo(x_clamped.dtype).eps
    left = torch.exp(-1.0 / x_clamped.clamp(min=machine_eps))
    right = torch.exp(-1.0 / (1.0 - x_clamped).clamp(min=machine_eps))
    interior = left / (left + right)
    weight = torch.where(
        x_clamped <= 0.0,
        torch.zeros_like(x_clamped),
        torch.where(x_clamped >= 1.0, torch.ones_like(x_clamped), interior),
    )
    q_pos = torch.where((q_neg * q_pos).sum(-1, keepdim=True) < 0.0, -q_pos, q_pos)
    blended = (1.0 - weight[..., None]) * q_neg + weight[..., None] * q_pos
    return blended / torch.sqrt((blended * blended).sum(-1, keepdim=True) + eps * eps)


__all__ = [
    "TileLangEdgeQuaternion",
    "edge_quaternion_reference",
    "tilelang_edge_quaternion",
]
