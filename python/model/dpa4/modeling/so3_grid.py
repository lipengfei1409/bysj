"""Independent PyTorch reference modules for SeZM SO(3) grid operations.

The implementation uses the packed ``(l, m)`` coefficient order and frame order
``[0, -1, +1, -2, +2, ...]`` from SeZM.  Projector matrices are derived
non-persistent buffers; no DeepMD module is imported at runtime.  Wigner blocks
reuse the local quaternion-polynomial reference in :mod:`.wigner`.

For the DPA4-Neo-OMol25-100M-v20260820 shape (``lmax=3, kmax=1``),
``SO3GridProjector`` has ``to_grid_mat.shape == (152, 48)`` and
``from_grid_mat.shape == (48, 152)``.  Its flattened projector input/output
signatures are ``[N,48,C] -> [N,152,C]`` and ``[N,152,C] -> [N,48,C]``.
``SO3GridNet`` uses canonical tensors ``[N,D,F,C]`` (or flat
``[N,D,F*C]``), where ``D=16`` and the two-focus message-node case has
``F=2``.  Its self-mode input channel width is ``2*K*C``; cross mode accepts
two separate ``K=3`` input tensors of width ``C`` and expands/contracts frames
internally.  The SO(3) frame count ``K`` is independent of focus count ``F``.

Lebedev precision 9 (38 sphere points, normalized weights) is embedded for the
checkpoint shape.  Other Lebedev rules can be passed as explicit tensors to the
projector constructor; this keeps the module independent of DeepMD's data loader.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn

from .wigner import _complex_wigner_block, _edge_quaternion, _quaternion_rotation_matrix


# Exact float64 values in the locally packaged Lebedev precision-9 rule.
_LEBEDEV9_POINTS = (
    (1.0, 0.0, 6.123233995736766e-17),
    (-1.0, 1.2246467991473532e-16, 6.123233995736766e-17),
    (6.123233995736766e-17, 1.0, 6.123233995736766e-17),
    (6.123233995736766e-17, -1.0, 6.123233995736766e-17),
    (0.0, 0.0, 1.0),
    (7.498798913309288e-33, 1.2246467991473532e-16, -1.0),
    (0.57735026918962584, 0.57735026918962573, 0.57735026918962573),
    (0.57735026918962584, 0.57735026918962573, -0.57735026918962573),
    (0.57735026918962584, -0.57735026918962573, 0.57735026918962573),
    (0.57735026918962584, -0.57735026918962573, -0.57735026918962573),
    (-0.57735026918962573, 0.57735026918962584, 0.57735026918962573),
    (-0.57735026918962573, 0.57735026918962584, -0.57735026918962573),
    (-0.57735026918962573, -0.57735026918962584, 0.57735026918962573),
    (-0.57735026918962573, -0.57735026918962584, -0.57735026918962573),
    (0.4597008433809831, 0.88807383397711526, 6.123233995736766e-17),
    (0.4597008433809831, -0.88807383397711526, 6.123233995736766e-17),
    (-0.45970084338098316, 0.88807383397711526, 6.123233995736766e-17),
    (-0.45970084338098316, -0.88807383397711526, 6.123233995736766e-17),
    (0.88807383397711526, 0.45970084338098305, 6.123233995736766e-17),
    (0.88807383397711526, -0.45970084338098305, 6.123233995736766e-17),
    (-0.88807383397711548, 0.45970084338098255, 6.123233995736766e-17),
    (-0.88807383397711548, -0.45970084338098255, 6.123233995736766e-17),
    (0.45970084338098305, 0.0, 0.88807383397711526),
    (0.45970084338098255, 0.0, -0.88807383397711548),
    (-0.45970084338098305, 5.6297116641185966e-17, 0.88807383397711526),
    (-0.45970084338098255, 5.6297116641185904e-17, -0.88807383397711548),
    (0.88807383397711526, 0.0, 0.4597008433809831),
    (0.88807383397711526, 0.0, -0.45970084338098316),
    (-0.88807383397711526, 1.0875767781865922e-16, 0.4597008433809831),
    (-0.88807383397711526, 1.0875767781865922e-16, -0.45970084338098316),
    (2.8148558320592983e-17, 0.45970084338098305, 0.88807383397711526),
    (2.8148558320592952e-17, 0.45970084338098255, -0.88807383397711548),
    (2.8148558320592983e-17, -0.45970084338098305, 0.88807383397711526),
    (2.8148558320592952e-17, -0.45970084338098255, -0.88807383397711548),
    (5.437883890932961e-17, 0.88807383397711526, 0.4597008433809831),
    (5.437883890932961e-17, 0.88807383397711526, -0.45970084338098316),
    (5.437883890932961e-17, -0.88807383397711526, 0.4597008433809831),
    (5.437883890932961e-17, -0.88807383397711526, -0.45970084338098316),
)
_LEBEDEV9_WEIGHTS = (
    0.0095238095238098838, 0.0095238095238098838, 0.0095238095238098838,
    0.0095238095238098838, 0.0095238095238098838, 0.0095238095238098838,
    0.032142857142856605, 0.032142857142856605, 0.032142857142856605,
    0.032142857142856605, 0.032142857142856605, 0.032142857142856605,
    0.032142857142856605, 0.032142857142856605,
    0.02857142857142865, 0.02857142857142865, 0.02857142857142865,
    0.02857142857142865, 0.02857142857142865, 0.02857142857142865,
    0.02857142857142865, 0.02857142857142865, 0.02857142857142865,
    0.02857142857142865, 0.02857142857142865, 0.02857142857142865,
    0.02857142857142865, 0.02857142857142865, 0.02857142857142865,
    0.02857142857142865, 0.02857142857142865, 0.02857142857142865,
    0.02857142857142865, 0.02857142857142865, 0.02857142857142865,
    0.02857142857142865, 0.02857142857142865, 0.02857142857142865,
)


def _packed_index(ell: int, order: int) -> int:
    """Packed index ``l*l + l + m`` for a valid ``(l,m)`` pair."""
    return ell * ell + ell + order


def _frame_set(kmax: int) -> tuple[int, ...]:
    return (0, *(order for k in range(1, kmax + 1) for order in (-k, k)))


def _frame_degree_index(lmax: int, mmax: int) -> torch.Tensor:
    return torch.tensor(
        [ell for ell in range(lmax + 1) for _ in range(2 * min(ell, mmax) + 1)],
        dtype=torch.long,
    )


def _quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for ``(w,x,y,z)`` quaternions."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def _wigner_grid(quaternion: torch.Tensor, lmax: int) -> torch.Tensor:
    """Evaluate packed SeZM real Wigner matrices for arbitrary quaternions."""
    q = quaternion.to(torch.float64)
    size = (lmax + 1) ** 2
    result = q.new_zeros((q.shape[0], size, size))
    result[:, 0, 0] = 1.0
    if lmax >= 1:
        # Upstream's Cartesian l=1 basis: permute x/y/z and apply its signs.
        perm = torch.tensor((1, 2, 0), dtype=torch.long, device=q.device)
        signs = q.new_tensor((-1.0, -1.0, 1.0))
        rotation = _quaternion_rotation_matrix(q)
        result[:, 1:4, 1:4] = (
            rotation.index_select(-2, perm).index_select(-1, perm)
            * signs[:, None] * signs[None, :]
        )
    for ell in range(2, lmax + 1):
        start, stop = ell * ell, (ell + 1) ** 2
        result[:, start:stop, start:stop] = _complex_wigner_block(q, ell)
    # SO3GridProjector stores row=m, column=k (the local-to-grid convention).
    return result.transpose(-1, -2).contiguous()


class SO3GridProjector(nn.Module):
    """Fixed SO(3) coefficient/grid transforms for packed ``(l,m,k)`` slots.

    Args:
        lmax: Maximum spherical degree.
        mmax: Maximum retained ``|m|``; defaults to ``lmax``.
        kmax: Maximum frame order. Frame slots are ordered ``[0,-1,+1,...]``.
        dtype: Floating dtype of derived projector matrices.
        lebedev_precision: Algebraic precision; the embedded default is 9.
        lebedev_points: Optional Cartesian unit points ``[A,3]`` for another rule.
        lebedev_weights: Optional normalized sphere weights ``[A]`` summing to one.

    ``to_grid_mat`` is ``[G, D*K]`` and ``from_grid_mat`` is ``[D*K, G]``.
    For ``lmax=3,mmax=3,kmax=1`` this is ``[152,48]`` and ``[48,152]``.
    The buffers and frame/index tables are non-persistent because checkpoint
    state contains learned tensors only.
    """

    def __init__(
        self,
        *,
        lmax: int,
        dtype: torch.dtype = torch.float32,
        mmax: int | None = None,
        kmax: int = 1,
        lebedev_precision: int | None = None,
        lebedev_points: torch.Tensor | None = None,
        lebedev_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lmax = int(lmax)
        self.mmax = self.lmax if mmax is None else int(mmax)
        self.kmax = int(kmax)
        if self.lmax < 0 or self.kmax < 0:
            raise ValueError("lmax and kmax must be nonnegative")
        if not 0 <= self.mmax <= self.lmax:
            raise ValueError("mmax must satisfy 0 <= mmax <= lmax")
        if lebedev_points is None and lebedev_weights is None:
            if lebedev_precision not in (None, 9):
                raise ValueError("pass points and weights for non-precision-9 rules")
            if self.lmax > 3:
                raise ValueError("precision-9 data supports lmax <= 3; pass a higher rule")
            lebedev_precision = 9
            points = torch.tensor(_LEBEDEV9_POINTS, dtype=torch.float64)
            weights = torch.tensor(_LEBEDEV9_WEIGHTS, dtype=torch.float64)
        elif lebedev_points is None or lebedev_weights is None:
            raise ValueError("lebedev_points and lebedev_weights must be supplied together")
        else:
            if lebedev_precision is None:
                raise ValueError("lebedev_precision is required with custom quadrature data")
            points = torch.as_tensor(lebedev_points, dtype=torch.float64, device="cpu")
            weights = torch.as_tensor(lebedev_weights, dtype=torch.float64, device="cpu")
        if points.ndim != 2 or points.shape[1] != 3 or weights.shape != (points.shape[0],):
            raise ValueError("Lebedev data must have shapes [A,3] and [A]")
        if points.shape[0] == 0 or not torch.isfinite(points).all() or not torch.isfinite(weights).all():
            raise ValueError("Lebedev data must be nonempty and finite")
        weights = weights / weights.sum()

        self.lebedev_precision = int(lebedev_precision)
        self.lebedev_npoints = int(points.shape[0])
        self.n_gamma = 1 if self.kmax == 0 else 3 * self.kmax + 1
        self.frame_set = _frame_set(self.kmax)
        self.frame_zero_index = 0
        self.n_frames = len(self.frame_set)
        self.packed_dim = (self.lmax + 1) ** 2
        self.coefficient_indices = tuple(
            _packed_index(ell, m)
            for ell in range(self.lmax + 1)
            for m in range(-min(ell, self.mmax), min(ell, self.mmax) + 1)
        )
        self.coeff_dim = len(self.coefficient_indices) * self.n_frames
        self.grid_size = self.lebedev_npoints * self.n_gamma
        to_grid, from_grid = self._build_projection_mats(points, weights)
        self.register_buffer("to_grid_mat", to_grid.to(dtype=dtype), persistent=False)
        self.register_buffer("from_grid_mat", from_grid.to(dtype=dtype), persistent=False)
        self.register_buffer(
            "frame_values", torch.tensor(self.frame_set, dtype=torch.long), persistent=False
        )

    def _build_projection_mats(
        self, points: torch.Tensor, weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = torch.arange(self.n_gamma, dtype=torch.float64) * (
            2.0 * math.pi / float(self.n_gamma)
        )
        half = gamma * 0.5
        gamma_q = torch.stack((torch.cos(half), torch.zeros_like(half),
                               torch.zeros_like(half), torch.sin(half)), dim=-1)
        edge_q = _edge_quaternion(points, 1.0e-14)
        edge_q = edge_q.repeat_interleave(self.n_gamma, dim=0)
        gamma_q = gamma_q.repeat(points.shape[0], 1)
        quaternion = _quaternion_multiply(gamma_q, edge_q)
        wigner = _wigner_grid(quaternion, self.lmax)
        haar_weight = weights.repeat_interleave(self.n_gamma) / float(self.n_gamma)

        to_grid = torch.zeros(self.grid_size, self.coeff_dim, dtype=torch.float64)
        from_grid = torch.zeros(self.coeff_dim, self.grid_size, dtype=torch.float64)
        coeff_position = {packed: i for i, packed in enumerate(self.coefficient_indices)}
        for ell in range(self.lmax + 1):
            for m in range(-min(ell, self.mmax), min(ell, self.mmax) + 1):
                position = coeff_position[_packed_index(ell, m)]
                row = _packed_index(ell, m)
                for frame_pos, k in enumerate(self.frame_set):
                    if abs(k) > ell:
                        continue
                    flat = position * self.n_frames + frame_pos
                    values = wigner[:, row, _packed_index(ell, k)]
                    to_grid[:, flat] = values
                    from_grid[flat, :] = (2 * ell + 1) * haar_weight * values
        return to_grid, from_grid

    def to_grid(self, coeff: torch.Tensor) -> torch.Tensor:
        """Map flattened coefficients ``[N,D*K,C]`` to grid values ``[N,G,C]``."""
        if coeff.ndim != 3 or coeff.shape[1] != self.coeff_dim:
            raise ValueError(f"coeff must have shape [N,{self.coeff_dim},C]")
        matrix = self.to_grid_mat.to(device=coeff.device, dtype=coeff.dtype)
        return torch.einsum("gj,njc->ngc", matrix, coeff)

    def from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Map grid values ``[N,G,C]`` to flattened coefficients ``[N,D*K,C]``."""
        if grid.ndim != 3 or grid.shape[1] != self.grid_size:
            raise ValueError(f"grid must have shape [N,{self.grid_size},C]")
        matrix = self.from_grid_mat.to(device=grid.device, dtype=grid.dtype)
        return torch.einsum("jg,ngc->njc", matrix, grid)


class ChannelLinear(nn.Module):
    """Checkpoint-shaped channel map with a ``[in_channels,out_channels]`` weight."""

    def __init__(
        self, *, in_channels: int, out_channels: int, dtype: torch.dtype,
        bias: bool = False, trainable: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, dtype=dtype))
        nn.init.uniform_(self.weight, -1.0 / math.sqrt(in_channels), 1.0 / math.sqrt(in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype)) if bias else None
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = torch.einsum("...i,io->...o", value, self.weight)
        return result + self.bias if self.bias is not None else result


class FocusLinear(nn.Module):
    """Focus-specific map; ``weight`` is stored as ``[Cin,F*Cout]``."""

    def __init__(
        self, *, in_channels: int, out_channels: int, n_focus: int,
        dtype: torch.dtype, bias: bool, trainable: bool, init_std: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_focus = int(n_focus)
        self.weight = nn.Parameter(torch.empty(in_channels, n_focus * out_channels, dtype=dtype))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        self.bias = nn.Parameter(torch.zeros(n_focus * out_channels, dtype=dtype)) if bias else None
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(self.in_channels, self.n_focus, self.out_channels)
        result = torch.einsum("nfi,ifo->nfo", value, weight)
        if self.bias is not None:
            result = result + self.bias.reshape(1, self.n_focus, self.out_channels)
        return result


def _project_frames(coeff: torch.Tensor, projection: ChannelLinear, n_frames: int) -> torch.Tensor:
    n, d, f, packed_channels = coeff.shape
    if packed_channels % n_frames:
        raise ValueError("packed channel width must be divisible by the frame count")
    per_frame = coeff.reshape(n, d, f, n_frames, packed_channels // n_frames)
    return projection(per_frame).reshape(n, d, f, -1)


class GridProduct(nn.Module):
    """Parameter-free pointwise grid product followed by SO(3) analysis."""

    def forward(
        self, left: torch.Tensor, right: torch.Tensor, *,
        to_grid: Callable[[torch.Tensor], torch.Tensor],
        from_grid: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        return from_grid(to_grid(left) * to_grid(right))


class GridMLP(nn.Module):
    """Independent polynomial grid MLP with self/cross-specific parameter shapes."""

    def __init__(
        self, *, channels: int, mode: str, n_frames: int, dtype: torch.dtype,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = str(mode).lower()
        self.n_frames = int(n_frames)
        if self.mode not in {"self", "cross"}:
            raise ValueError("mode must be 'self' or 'cross'")
        input_channels = 2 * self.channels if self.mode == "self" else self.channels
        self.hidden_channels = 2 * self.channels
        self.left_proj = ChannelLinear(
            in_channels=input_channels, out_channels=self.hidden_channels,
            dtype=dtype, trainable=trainable,
        )
        self.right_proj = ChannelLinear(
            in_channels=input_channels, out_channels=self.hidden_channels,
            dtype=dtype, trainable=trainable,
        )
        self.out_proj = ChannelLinear(
            in_channels=self.hidden_channels, out_channels=self.channels,
            dtype=dtype, trainable=trainable,
        )

    def forward(self, left, right, *, to_grid, from_grid):
        if self.mode == "self":
            n, d, f, packed = left.shape
            frame_shape = (n, d, f, self.n_frames, packed // self.n_frames)
            fused = torch.cat((left.reshape(frame_shape), right.reshape(frame_shape)), dim=-1)
            fused = fused.reshape(n, d, f, -1)
            left = _project_frames(fused, self.left_proj, self.n_frames)
            right = _project_frames(fused, self.right_proj, self.n_frames)
        else:
            left = _project_frames(left, self.left_proj, self.n_frames)
            right = _project_frames(right, self.right_proj, self.n_frames)
        product_coeff = from_grid(to_grid(left) * to_grid(right))
        return _project_frames(product_coeff, self.out_proj, self.n_frames)


class GridBranch(nn.Module):
    """Scalar-routed parallel quadratic branches; the one-branch router is retained."""

    def __init__(
        self, *, channels: int, n_branches: int, n_frames: int,
        dtype: torch.dtype, trainable: bool,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.n_branches = int(n_branches)
        self.n_frames = int(n_frames)
        if self.n_branches < 1:
            raise ValueError("n_branches must be positive")
        self.left_proj = ChannelLinear(
            in_channels=self.channels, out_channels=self.n_branches * self.channels,
            dtype=dtype, trainable=trainable,
        )
        self.right_proj = ChannelLinear(
            in_channels=self.channels, out_channels=self.n_branches * self.channels,
            dtype=dtype, trainable=trainable,
        )
        self.router = ChannelLinear(
            in_channels=2 * self.channels, out_channels=self.n_branches,
            dtype=dtype, trainable=trainable and self.n_branches > 1,
        )
        self.out_proj = ChannelLinear(
            in_channels=self.channels, out_channels=self.channels,
            dtype=dtype, trainable=trainable,
        )

    def forward(self, left, right, scalar_pair, *, to_grid, from_grid):
        left = _project_frames(left, self.left_proj, self.n_frames)
        right = _project_frames(right, self.right_proj, self.n_frames)
        branch_grid = to_grid(left) * to_grid(right)
        n, g, f, wide = branch_grid.shape
        branch_grid = branch_grid.reshape(n, g, f, self.n_branches, self.channels)
        route = torch.softmax(self.router(scalar_pair), dim=-1)
        mixed_grid = torch.einsum("ngfbc,nfb->ngfc", branch_grid, route)
        coeff = from_grid(mixed_grid)
        return _project_frames(coeff, self.out_proj, self.n_frames)


class FrameExpand(nn.Module):
    """Per-degree expansion ``[N,D,F,C] -> [N,D,F,K*C]`` for cross nets."""

    def __init__(self, *, lmax: int, mmax: int, n_frames: int, channels: int,
                 dtype: torch.dtype, trainable: bool) -> None:
        super().__init__()
        self.n_frames, self.channels = int(n_frames), int(channels)
        self.register_buffer("degree_index", _frame_degree_index(lmax, mmax), persistent=False)
        self.weight = nn.Parameter(torch.empty(lmax + 1, channels, n_frames * channels, dtype=dtype))
        nn.init.uniform_(self.weight, -1.0 / math.sqrt(channels), 1.0 / math.sqrt(channels))
        self.weight.requires_grad_(trainable)

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        weight = self.weight.index_select(0, self.degree_index)
        return torch.einsum("ndfi,dio->ndfo", coeff, weight)


class FrameContract(nn.Module):
    """Per-degree contraction ``[N,D,F,K*C] -> [N,D,F,C]`` for cross nets."""

    def __init__(self, *, lmax: int, mmax: int, n_frames: int, channels: int,
                 dtype: torch.dtype, trainable: bool) -> None:
        super().__init__()
        self.n_frames, self.channels = int(n_frames), int(channels)
        self.register_buffer("degree_index", _frame_degree_index(lmax, mmax), persistent=False)
        self.weight = nn.Parameter(torch.empty(lmax + 1, n_frames * channels, channels, dtype=dtype))
        nn.init.uniform_(self.weight, -1.0 / math.sqrt(n_frames * channels), 1.0 / math.sqrt(n_frames * channels))
        self.weight.requires_grad_(trainable)

    def forward(self, coeff: torch.Tensor) -> torch.Tensor:
        weight = self.weight.index_select(0, self.degree_index)
        return torch.einsum("ndfi,dio->ndfo", coeff, weight)

    def forward_scalar(self, coeff: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ndfi,dio->ndfo", coeff, self.weight[:1])


class _SwiGLU(nn.Module):
    """Scalar SwiGLU: first half is gate, second half is value."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, branch = value.chunk(2, dim=-1)
        return gate * torch.sigmoid(gate) * branch


class SO3GridNet(nn.Module):
    """SO(3) grid operation for self (FFN) or cross (message-node) inputs.

    ``layout='ndfc'`` accepts ``[N,D,F,C]``; ``'flat'`` accepts ``[N,D,F*C]``.
    Self mode takes one tensor with width ``2*K*C``. Cross mode takes query and
    context tensors with width ``C`` each, internally applies frame expansion,
    grid operation and frame contraction. Scalar gating always reads the
    ``l=0,k=0`` coefficients and the separate scalar SwiGLU branch is added to
    that slot. ``forward_scalar`` returns the scalar coefficient only.
    """

    def __init__(
        self, *, lmax: int, channels: int, n_focus: int = 1,
        mode: str, op_type: str, dtype: torch.dtype,
        layout: str = "ndfc", mmax: int | None = None, kmax: int = 1,
        lebedev_precision: int | None = None,
        lebedev_points: torch.Tensor | None = None,
        lebedev_weights: torch.Tensor | None = None,
        grid_branches: int = 1, mlp_bias: bool = False,
        residual_scale_init: float | None = None, trainable: bool = True,
    ) -> None:
        super().__init__()
        self.lmax, self.channels, self.n_focus = int(lmax), int(channels), int(n_focus)
        self.mode, self.op_type = str(mode).lower(), str(op_type).lower()
        self.layout = str(layout).lower()
        if self.mode not in {"self", "cross"}:
            raise ValueError("mode must be 'self' or 'cross'")
        if self.op_type not in {"glu", "mlp", "branch"}:
            raise ValueError("op_type must be 'glu', 'mlp' or 'branch'")
        if self.layout not in {"ndfc", "nfdc", "fndc", "flat"}:
            raise ValueError("unsupported layout")
        if self.mode == "self" and self.layout == "flat":
            raise ValueError("flat layout is only supported for cross mode")
        self.projector = SO3GridProjector(
            lmax=lmax, mmax=mmax, kmax=kmax, dtype=dtype,
            lebedev_precision=lebedev_precision,
            lebedev_points=lebedev_points, lebedev_weights=lebedev_weights,
        )
        self.n_frames = self.projector.n_frames
        self.n_coeff = self.projector.coeff_dim // self.n_frames
        self.expanded_channels = self.n_frames * self.channels
        self.frame_zero_index = self.projector.frame_zero_index
        self.dtype = dtype
        self.frame_expand = None
        self.frame_contract = None
        if self.mode == "cross":
            self.frame_expand = FrameExpand(
                lmax=lmax, mmax=self.projector.mmax, n_frames=self.n_frames,
                channels=channels, dtype=dtype, trainable=trainable,
            )
            self.frame_contract = FrameContract(
                lmax=lmax, mmax=self.projector.mmax, n_frames=self.n_frames,
                channels=channels, dtype=dtype, trainable=trainable,
            )
        self.scalar_act = _SwiGLU()
        self.scalar_gate = FocusLinear(
            in_channels=2 * channels, out_channels=channels, n_focus=n_focus,
            dtype=dtype, bias=mlp_bias, trainable=trainable,
        )
        if self.op_type == "glu":
            self.grid_op = GridProduct()
        elif self.op_type == "mlp":
            self.grid_op = GridMLP(
                channels=channels, mode=self.mode, n_frames=self.n_frames,
                dtype=dtype, trainable=trainable,
            )
        else:
            self.grid_op = GridBranch(
                channels=channels, n_branches=grid_branches, n_frames=self.n_frames,
                dtype=dtype, trainable=trainable,
            )
        self.output_channels = channels if self.mode == "cross" else self.expanded_channels
        if residual_scale_init is None:
            self.register_parameter("residual_scale", None)
        else:
            self.residual_scale = nn.Parameter(torch.full(
                (self.n_focus, self.output_channels), float(residual_scale_init), dtype=dtype
            ))
            self.residual_scale.requires_grad_(trainable)

    def _to_ndfc(self, value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if self.layout == "ndfc":
            return value, tuple(value.shape)
        if self.layout == "nfdc":
            return value.transpose(1, 2), tuple(value.shape)
        if self.layout == "fndc":
            return value.permute(1, 2, 0, 3), tuple(value.shape)
        if value.ndim != 3:
            raise ValueError("flat layout requires [N,D,F*C]")
        n, d, packed = value.shape
        if packed % self.n_focus:
            raise ValueError("flat channel width must be divisible by n_focus")
        return value.reshape(n, d, self.n_focus, packed // self.n_focus), tuple(value.shape)

    def _restore_layout(self, value: torch.Tensor, shape_info: tuple[int, ...], scalar_only: bool):
        if self.layout == "ndfc":
            return value
        if self.layout == "nfdc":
            return value.transpose(1, 2)
        if self.layout == "fndc":
            return value.permute(2, 0, 1, 3)
        n, original_d, _ = shape_info
        d = 1 if scalar_only else original_d
        return value.reshape(n, d, value.shape[2] * value.shape[3])

    def _check_coeff(self, value: torch.Tensor, width: int, name: str) -> None:
        if value.ndim != 4 or value.shape[1] != self.n_coeff or value.shape[2] != self.n_focus:
            raise ValueError(f"{name} must have shape [N,{self.n_coeff},{self.n_focus},C]")
        if value.shape[-1] != width:
            raise ValueError(f"{name} channel width must be {width}, got {value.shape[-1]}")

    def _split_self(self, query: torch.Tensor):
        expected = 2 * self.expanded_channels
        self._check_coeff(query, expected, "query")
        return query.chunk(2, dim=-1)

    def _extract_scalar(self, coeff: torch.Tensor) -> torch.Tensor:
        n, _, f, packed = coeff.shape
        return coeff.reshape(n, coeff.shape[1], f, self.n_frames, packed // self.n_frames)[:, 0, :, self.frame_zero_index]

    def _to_grid(self, coeff: torch.Tensor) -> torch.Tensor:
        n, d, f, packed = coeff.shape
        if packed % self.n_frames:
            raise ValueError("coefficient channels must be divisible by frame count")
        channels = packed // self.n_frames
        coeff_view = coeff.reshape(n, d, f, self.n_frames, channels)
        matrix = self.projector.to_grid_mat.reshape(self.projector.grid_size, d, self.n_frames)
        matrix = matrix.to(device=coeff.device, dtype=coeff.dtype)
        return torch.einsum("gdk,ndfkc->ngfc", matrix, coeff_view)

    def _from_grid(self, grid: torch.Tensor) -> torch.Tensor:
        n, g, f, channels = grid.shape
        matrix = self.projector.from_grid_mat.reshape(self.n_coeff, self.n_frames, g)
        matrix = matrix.to(device=grid.device, dtype=grid.dtype)
        coeff = torch.einsum("dkg,ngfc->ndfkc", matrix, grid)
        return coeff.reshape(n, self.n_coeff, f, self.n_frames * channels)

    def _from_grid_scalar(self, grid: torch.Tensor) -> torch.Tensor:
        n, g, f, channels = grid.shape
        matrix = self.projector.from_grid_mat.reshape(self.n_coeff, self.n_frames, g)[:1]
        matrix = matrix.to(device=grid.device, dtype=grid.dtype)
        coeff = torch.einsum("dkg,ngfc->ndfkc", matrix, grid)
        return coeff.reshape(n, 1, f, self.n_frames * channels)

    def _scalar_path(self, coeff: torch.Tensor, scalar_pair: torch.Tensor) -> torch.Tensor:
        scalar_out = self.scalar_act(scalar_pair)
        gate = torch.sigmoid(self.scalar_gate(scalar_pair))
        n, d, f, packed = coeff.shape
        view = coeff.reshape(n, d, f, self.n_frames, self.channels)
        gated = view * gate[:, None, :, None, :]
        injection = torch.zeros_like(gated)
        injection[:, 0, :, self.frame_zero_index, :] = scalar_out
        return (gated + injection).reshape(n, d, f, packed)

    def _contract(self, coeff: torch.Tensor, scalar_only: bool) -> torch.Tensor:
        if self.frame_contract is None:
            return coeff
        if scalar_only:
            return self.frame_contract.forward_scalar(coeff)
        return self.frame_contract(coeff)

    def _forward(self, query: torch.Tensor, context: torch.Tensor | None, scalar_only: bool):
        input_dtype = query.dtype
        query_ndfc, shape_info = self._to_ndfc(query)
        if self.mode == "self":
            left, right = self._split_self(query_ndfc)
            scalar_pair = torch.cat((self._extract_scalar(left), self._extract_scalar(right)), dim=-1)
        else:
            if context is None:
                raise ValueError("context is required for cross mode")
            context_ndfc, _ = self._to_ndfc(context)
            self._check_coeff(query_ndfc, self.channels, "query")
            self._check_coeff(context_ndfc, self.channels, "context")
            if context_ndfc.shape[:3] != query_ndfc.shape[:3]:
                raise ValueError("query and context batch/coefficient/focus shapes must match")
            scalar_pair = torch.cat((query_ndfc[:, 0], context_ndfc[:, 0]), dim=-1)
            left, right = self.frame_expand(query_ndfc), self.frame_expand(context_ndfc)

        left, right, scalar_pair = left.to(self.dtype), right.to(self.dtype), scalar_pair.to(self.dtype)
        to_grid = self._to_grid
        from_grid = self._from_grid_scalar if scalar_only else self._from_grid
        if isinstance(self.grid_op, GridProduct):
            coeff = self.grid_op(left, right, to_grid=to_grid, from_grid=from_grid)
        elif isinstance(self.grid_op, GridMLP):
            coeff = self.grid_op(left, right, to_grid=to_grid, from_grid=from_grid)
        else:
            coeff = self.grid_op(left, right, scalar_pair, to_grid=to_grid, from_grid=from_grid)
        coeff = self._scalar_path(coeff, scalar_pair)
        coeff = self._contract(coeff, scalar_only)
        if self.residual_scale is not None:
            coeff = coeff * self.residual_scale.reshape(1, 1, self.n_focus, self.output_channels)
        return self._restore_layout(coeff.to(dtype=input_dtype), shape_info, scalar_only)

    def forward(self, query: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Return full packed coefficients (or flat cross-layout features)."""
        return self._forward(query, context, scalar_only=False)

    def forward_scalar(self, query: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        """Return only ``l=0``; signatures match :meth:`forward`."""
        return self._forward(query, context, scalar_only=True)


class SO3Linear(nn.Module):
    """Per-degree linear channel map with upstream-compatible state tensor shapes."""

    def __init__(
        self, *, lmax: int, in_channels: int, out_channels: int,
        n_focus: int = 1, dtype: torch.dtype, mlp_bias: bool = False,
        trainable: bool = True, init_std: float | None = None,
    ) -> None:
        super().__init__()
        self.lmax, self.in_channels = int(lmax), int(in_channels)
        self.out_channels, self.n_focus = int(out_channels), int(n_focus)
        self.weight = nn.Parameter(torch.empty(
            lmax + 1, in_channels, n_focus * out_channels, dtype=dtype
        ))
        if init_std is None:
            nn.init.uniform_(self.weight, -1.0 / math.sqrt(in_channels), 1.0 / math.sqrt(in_channels))
        else:
            nn.init.normal_(self.weight, mean=0.0, std=init_std)
        self.bias = nn.Parameter(torch.zeros(n_focus * out_channels, dtype=dtype)) if mlp_bias else None
        self.register_buffer(
            "expand_index", _frame_degree_index(lmax, lmax), persistent=True
        )
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(self.lmax + 1, self.in_channels, self.n_focus, self.out_channels)
        expanded = weight.index_select(0, self.expand_index)
        result = torch.einsum("ndfi,difo->ndfo", value, expanded)
        if self.bias is not None:
            result = result.clone()
            result[:, 0] = result[:, 0] + self.bias.reshape(self.n_focus, self.out_channels)
        return result

    def forward_scalar(self, value: torch.Tensor) -> torch.Tensor:
        weight = self.weight[0].reshape(self.in_channels, self.n_focus, self.out_channels)
        result = torch.einsum("ndfi,ifo->ndfo", value[:, :1], weight)
        if self.bias is not None:
            result = result + self.bias.reshape(1, self.n_focus, self.out_channels)
        return result


class EquivariantFFN(nn.Module):
    """SO3Linear -> self SO(3) grid activation -> SO3Linear feed-forward block.

    This independent reference implements the ``ffn_so3_grid=True`` path.  Its
    parameter names mirror the upstream composition. ``forward_scalar`` computes
    only degree zero for scalar-only readout while preserving the full input
    coefficient interactions.
    """

    def __init__(
        self, *, lmax: int, channels: int, hidden_channels: int,
        dtype: torch.dtype, kmax: int = 1, grid_mlp: bool = False,
        grid_branch: int = 0, ffn_so3_grid: bool = True,
        lebedev_precision: int | None = None,
        lebedev_points: torch.Tensor | None = None,
        lebedev_weights: torch.Tensor | None = None,
        mlp_bias: bool = False, trainable: bool = True,
    ) -> None:
        super().__init__()
        if not ffn_so3_grid:
            raise NotImplementedError("this reference EquivariantFFN implements the SO3-grid path")
        self.lmax, self.channels, self.hidden_channels = int(lmax), int(channels), int(hidden_channels)
        self.kmax = int(kmax)
        n_frames = 2 * self.kmax + 1
        self.so3_linear_1 = SO3Linear(
            lmax=lmax, in_channels=channels, out_channels=2 * n_frames * hidden_channels,
            dtype=dtype, mlp_bias=mlp_bias, trainable=trainable,
        )
        op_type = "branch" if grid_branch > 0 else ("mlp" if grid_mlp else "glu")
        self.act = SO3GridNet(
            lmax=lmax, kmax=kmax, channels=hidden_channels, n_focus=1,
            mode="self", op_type=op_type, dtype=dtype, layout="ndfc",
            lebedev_precision=lebedev_precision,
            lebedev_points=lebedev_points, lebedev_weights=lebedev_weights,
            grid_branches=max(1, int(grid_branch)), mlp_bias=mlp_bias,
            trainable=trainable,
        )
        self.so3_linear_2 = SO3Linear(
            lmax=lmax, in_channels=n_frames * hidden_channels, out_channels=channels,
            dtype=dtype, mlp_bias=mlp_bias, trainable=trainable, init_std=0.0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.so3_linear_2(self.act(self.so3_linear_1(value)))

    def forward_scalar(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.act.forward_scalar(self.so3_linear_1(value))
        return self.so3_linear_2.forward_scalar(hidden)
