"""Standalone TileLang l=1 Wigner-D blocks for DPA4 edge frames."""

from functools import lru_cache

import torch

_BLOCK_EDGES = 16


def _tilelang_modules():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as error:
        raise RuntimeError(
            "TileLang l=1 Wigner-D requires TileLang to be installed"
        ) from error
    return tilelang, T


def _rotation_entry(T, row, col, w, x, y, z):
    """Select one element of the active quaternion rotation matrix."""
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return T.if_then_else(
        row == 0,
        T.if_then_else(col == 0, r00, T.if_then_else(col == 1, r01, r02)),
        T.if_then_else(
            row == 1,
            T.if_then_else(col == 0, r10, T.if_then_else(col == 1, r11, r12)),
            T.if_then_else(col == 0, r20, T.if_then_else(col == 1, r21, r22)),
        ),
    )


def _rotation_derivative(T, row, col, component, w, x, y, z):
    """Select one quaternion derivative of a rotation-matrix entry."""
    if row == 0 and col == 0:
        values = (0.0, 0.0, -4.0 * y, -4.0 * z)
    elif row == 0 and col == 1:
        values = (-2.0 * z, 2.0 * y, 2.0 * x, -2.0 * w)
    elif row == 0 and col == 2:
        values = (2.0 * y, 2.0 * z, 2.0 * w, 2.0 * x)
    elif row == 1 and col == 0:
        values = (2.0 * z, 2.0 * y, 2.0 * x, 2.0 * w)
    elif row == 1 and col == 1:
        values = (0.0, -4.0 * x, 0.0, -4.0 * z)
    elif row == 1 and col == 2:
        values = (-2.0 * x, -2.0 * w, 2.0 * z, 2.0 * y)
    elif row == 2 and col == 0:
        values = (-2.0 * y, 2.0 * z, -2.0 * w, 2.0 * x)
    elif row == 2 and col == 1:
        values = (2.0 * x, 2.0 * w, 2.0 * z, 2.0 * y)
    else:
        values = (0.0, -4.0 * x, -4.0 * y, 0.0)
    return T.if_then_else(
        component == 0,
        values[0],
        T.if_then_else(
            component == 1,
            values[1],
            T.if_then_else(component == 2, values[2], values[3]),
        ),
    )


@lru_cache(maxsize=1)
def _forward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def wigner_l1_forward(n_edges, block_edges, eps, dtype=T.float32):
        @T.prim_func
        def main(
            quaternion: T.Tensor((n_edges, 4), dtype),
            d_full: T.Tensor((n_edges, 4, 4), dtype),
            dt_full: T.Tensor((n_edges, 4, 4), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid,):
                for edge_offset, row, col in T.Parallel(block_edges, 4, 4):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        qw = quaternion[edge, 0]
                        qx = quaternion[edge, 1]
                        qy = quaternion[edge, 2]
                        qz = quaternion[edge, 3]
                        norm = T.sqrt(qw * qw + qx * qx + qy * qy + qz * qz + eps * eps)
                        w = qw / norm
                        x = qx / norm
                        y = qy / norm
                        z = qz / norm
                        if row == 0 and col == 0:
                            value = 1.0
                        elif row == 0 or col == 0:
                            value = 0.0
                        else:
                            # SeZM's real l=1 basis permutes Cartesian axes [1, 2, 0]
                            # and applies sign [-1, -1, +1] on both matrix axes.
                            rrow = T.if_then_else(
                                row == 1, 1, T.if_then_else(row == 2, 2, 0)
                            )
                            rcol = T.if_then_else(
                                col == 1, 1, T.if_then_else(col == 2, 2, 0)
                            )
                            sign_row = T.if_then_else(row == 3, 1.0, -1.0)
                            sign_col = T.if_then_else(col == 3, 1.0, -1.0)
                            value = (
                                sign_row
                                * sign_col
                                * _rotation_entry(T, rrow, rcol, w, x, y, z)
                            )
                        d_full[edge, row, col] = value
                        dt_full[edge, col, row] = value

        return main

    return wigner_l1_forward


@lru_cache(maxsize=1)
def _backward_factory():
    tilelang, T = _tilelang_modules()

    @tilelang.jit(target="hip")
    def wigner_l1_backward(n_edges, block_edges, eps, dtype=T.float32):
        @T.prim_func
        def main(
            grad_d: T.Tensor((n_edges, 4, 4), dtype),
            grad_dt: T.Tensor((n_edges, 4, 4), dtype),
            quaternion: T.Tensor((n_edges, 4), dtype),
            grad_quaternion: T.Tensor((n_edges, 4), dtype),
        ):
            with T.Kernel(T.ceildiv(n_edges, block_edges), threads=128) as (pid,):
                for edge_offset in T.Parallel(block_edges):
                    edge = pid * block_edges + edge_offset
                    if edge < n_edges:
                        qw = quaternion[edge, 0]
                        qx = quaternion[edge, 1]
                        qy = quaternion[edge, 2]
                        qz = quaternion[edge, 3]
                        norm = T.sqrt(qw * qw + qx * qx + qy * qy + qz * qz + eps * eps)
                        w = qw / norm
                        x = qx / norm
                        y = qy / norm
                        z = qz / norm
                        gw = 0.0
                        gx = 0.0
                        gy = 0.0
                        gz = 0.0
                        inverse_perm = (2, 0, 1)
                        signs = (-1.0, -1.0, 1.0)
                        for rrow in T.static_range(0, 3):
                            for rcol in T.static_range(0, 3):
                                i = inverse_perm[rrow] + 1
                                j = inverse_perm[rcol] + 1
                                sign = signs[i - 1] * signs[j - 1]
                                g = sign * (grad_d[edge, i, j] + grad_dt[edge, j, i])
                                gw += g * _rotation_derivative(
                                    T, rrow, rcol, 0, w, x, y, z
                                )
                                gx += g * _rotation_derivative(
                                    T, rrow, rcol, 1, w, x, y, z
                                )
                                gy += g * _rotation_derivative(
                                    T, rrow, rcol, 2, w, x, y, z
                                )
                                gz += g * _rotation_derivative(
                                    T, rrow, rcol, 3, w, x, y, z
                                )
                        dot = gw * qw + gx * qx + gy * qy + gz * qz
                        inv_norm = 1.0 / norm
                        inv_norm3 = inv_norm * inv_norm * inv_norm
                        grad_quaternion[edge, 0] = gw * inv_norm - qw * dot * inv_norm3
                        grad_quaternion[edge, 1] = gx * inv_norm - qx * dot * inv_norm3
                        grad_quaternion[edge, 2] = gy * inv_norm - qy * dot * inv_norm3
                        grad_quaternion[edge, 3] = gz * inv_norm - qz * dot * inv_norm3

        return main

    return wigner_l1_backward


def _check_inputs(quaternion: torch.Tensor) -> None:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion must have shape (n_edges, 4)")
    if quaternion.dtype != torch.float32:
        raise TypeError("TileLang l=1 Wigner-D requires float32 input")
    if not quaternion.is_cuda or torch.version.hip is None:
        raise RuntimeError("TileLang l=1 Wigner-D requires a HIP/DCU tensor")


def _forward(quaternion: torch.Tensor, eps: float):
    _check_inputs(quaternion)
    n_edges = quaternion.shape[0]
    d_full = torch.empty(
        (n_edges, 4, 4), dtype=quaternion.dtype, device=quaternion.device
    )
    dt_full = torch.empty_like(d_full)
    if n_edges:
        _, T = _tilelang_modules()
        kernel = _forward_factory()(n_edges, _BLOCK_EDGES, eps, T.float32)
        kernel(quaternion.contiguous(), d_full, dt_full)
    return d_full, dt_full


def _backward(grad_d, grad_dt, quaternion: torch.Tensor, eps: float):
    grad_quaternion = torch.empty_like(quaternion)
    if quaternion.shape[0]:
        _, T = _tilelang_modules()
        kernel = _backward_factory()(quaternion.shape[0], _BLOCK_EDGES, eps, T.float32)
        kernel(
            grad_d.contiguous(),
            grad_dt.contiguous(),
            quaternion.contiguous(),
            grad_quaternion,
        )
    return grad_quaternion


class TileLangWignerL1(torch.autograd.Function):
    """Compute D and D-transpose through separate TileLang kernels."""

    @staticmethod
    def forward(ctx, quaternion: torch.Tensor, eps: float = 1.0e-7):
        _check_inputs(quaternion)
        quaternion = quaternion.contiguous()
        ctx.eps = float(eps)
        ctx.save_for_backward(quaternion)
        return _forward(quaternion, ctx.eps)

    @staticmethod
    def backward(ctx, grad_d, grad_dt):
        (quaternion,) = ctx.saved_tensors
        if grad_d is None:
            grad_d = torch.zeros(
                (quaternion.shape[0], 4, 4),
                dtype=quaternion.dtype,
                device=quaternion.device,
            )
        if grad_dt is None:
            grad_dt = torch.zeros_like(grad_d)
        return _backward(grad_d, grad_dt, quaternion, ctx.eps), None


def tilelang_wigner_l1(
    quaternion: torch.Tensor, eps: float = 1.0e-7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the packed l=0/l=1 Wigner-D pair for DPA4."""
    return TileLangWignerL1.apply(quaternion, float(eps))


def wigner_l1_reference(
    quaternion: torch.Tensor, eps: float = 1.0e-7
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable PyTorch reference for the SeZM packed l=1 basis."""
    norm = torch.sqrt((quaternion * quaternion).sum(dim=-1, keepdim=True) + eps * eps)
    q = quaternion / norm
    w, x, y, z = q.unbind(dim=-1)
    rotation = torch.stack(
        [
            torch.stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    perm = torch.tensor([1, 2, 0], device=quaternion.device)
    signs = quaternion.new_tensor([-1.0, -1.0, 1.0])
    l1 = rotation.index_select(-2, perm).index_select(-1, perm)
    l1 = l1 * signs[None, :, None] * signs[None, None, :]
    d_full = quaternion.new_zeros((quaternion.shape[0], 4, 4))
    d_full[:, 0, 0] = 1.0
    d_full[:, 1:, 1:] = l1
    return d_full, d_full.transpose(-1, -2)


__all__ = ["TileLangWignerL1", "tilelang_wigner_l1", "wigner_l1_reference"]
