"""Parameter-owning numerical components; no dependency on DeePMD imports."""
from __future__ import annotations

import math

import torch
from torch import nn

from dcu_dpa4 import ops
from dcu_dpa4 import blas


class ChannelLinear(nn.Module):
    """DeePMD matrix layout [K,C], arbitrary leading axes, explicit backend."""
    def __init__(self, in_features, out_features, *, backend='reference', bias=False):
        super().__init__()
        if backend not in ('reference', 'hip', 'rocblas'):
            raise ValueError('linear backend must be reference, hip or rocblas')
        if min(in_features, out_features) <= 0:
            raise ValueError('linear dimensions must be positive')
        self.backend = backend
        self.matrix = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.normal_(self.matrix, std=1 / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        flat = x.reshape(math.prod(x.shape[:-1]), x.shape[-1])
        if self.backend == 'rocblas':
            result = blas.linear(flat, self.matrix, self.bias, backend='rocblas')
        else:
            result = ops.linear(flat, self.matrix, self.bias, backend=self.backend)
        return result.reshape(*x.shape[:-1], self.matrix.shape[1])


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels, *, eps=1e-7, backend='reference'):
        super().__init__()
        self.adam_scale = nn.Parameter(torch.ones(channels))
        self.eps, self.backend = eps, backend

    def forward(self, x):
        return ops.rmsnorm(x, self.adam_scale, self.eps, backend=self.backend)


def activation(name):
    if name == 'gelu':
        return nn.GELU(approximate='tanh')
    functions = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'gelu': nn.GELU, 'relu': nn.ReLU}
    if name not in functions:
        raise NotImplementedError(f'activation {name!r} is not supported')
    return functions[name]()


class RadialMLP(nn.Module):
    """Linear → optional ordinary RMSNorm → activation; last layer linear.

    This is SeZM RadialMLP, not the fitting network (no residual/bias/idt).
    Parameter names preserve its ``net.<index>.matrix/adam_scale`` layout.
    """
    def __init__(self, widths, *, lmax, channels, radial_norm=True, activation_function='silu',
                 eps=1e-7, linear_backend='reference', norm_backend='reference'):
        super().__init__()
        if len(widths) < 2 or min(widths) <= 0 or widths[-1] != (lmax + 1) * channels:
            raise ValueError('radial widths must end in (lmax+1)*channels')
        self.lmax, self.channels = lmax, channels
        layers = []
        for i, (ni, no) in enumerate(zip(widths, widths[1:])):
            layers.append(ChannelLinear(ni, no, backend=linear_backend))
            if i < len(widths) - 2:
                if radial_norm:
                    layers.append(ChannelRMSNorm(no, eps=eps, backend=norm_backend))
                layers.append(activation(activation_function))
        self.net = nn.Sequential(*layers)

    def forward(self, rbf, env=None):
        return {'y': self.net(rbf).reshape(rbf.shape[0], self.lmax + 1, self.channels)}


class TypeEmbedding(nn.Module):
    def __init__(self, table):
        super().__init__()
        if table.ndim != 2 or not table.is_floating_point():
            raise ValueError('type table must be floating [T+1,C]')
        self.adam_type_embedding = nn.Parameter(table.detach().clone())

    def forward(self, atype):
        return {'y': torch.embedding(self.adam_type_embedding, atype)}


class ChargeSpinEmbedding(nn.Module):
    """Frame-level charge/spin embedding used by OMol-style SeZM models."""

    def __init__(self, charge_table, spin_table, matrix, bias, *, activation_function='silu', backend='reference'):
        super().__init__()
        if charge_table.ndim != 2 or charge_table.shape[0] != 200:
            raise ValueError('charge embedding table must have shape [200,C]')
        if spin_table.ndim != 2 or spin_table.shape[0] != 100 or spin_table.shape[1] != charge_table.shape[1]:
            raise ValueError('spin embedding table must have shape [100,C]')
        channels = charge_table.shape[1]
        if matrix.shape != (2 * channels, channels) or bias.shape != (channels,):
            raise ValueError('charge/spin mixing layer has incompatible shapes')
        if any(t.dtype != charge_table.dtype for t in (spin_table, matrix, bias)):
            raise TypeError('charge/spin parameters must share a dtype')
        self.charge_embedding = TypeEmbedding(charge_table)
        self.spin_embedding = TypeEmbedding(spin_table)
        self.mix_layer = ChannelLinear(2 * channels, channels, backend=backend, bias=True)
        with torch.no_grad():
            self.mix_layer.matrix.copy_(matrix)
            self.mix_layer.bias.copy_(bias)
        self.activate = activation(activation_function)

    def forward(self, charge_spin):
        if charge_spin.ndim != 2 or charge_spin.shape[1] != 2:
            raise ValueError('charge_spin must have shape [F,2]')
        charge = charge_spin[:, 0].to(dtype=torch.int64) + 100
        spin = charge_spin[:, 1].to(dtype=torch.int64)
        charge_embedding = self.charge_embedding(charge)['y']
        spin_embedding = self.spin_embedding(spin)['y']
        mixed = self.mix_layer(torch.cat((charge_embedding, spin_embedding), dim=-1))
        return {'y': self.activate(mixed)}


class EnvironmentSeed(nn.Module):
    """Checkpoint-shaped environment matrix that produces FiLM logits."""

    def __init__(self, n_types, n_radial, channels, *, env_type_table, activation_function='silu', backend='reference', bias=False, eps=1e-7):
        super().__init__()
        embed_dim = min(channels, 128)
        axis_dim = min(4 if embed_dim < 64 else 8, max(1, embed_dim - 1))
        type_dim = min(32, max(8, channels // 4))
        rbf_out_dim = max(32, embed_dim - 2 * type_dim)
        hidden_dim = min(256, max(2 * embed_dim, rbf_out_dim + 2 * type_dim))
        if env_type_table.shape != (n_types + 1, type_dim):
            raise ValueError('environment type table shape does not match the checkpoint config')
        self.embed_dim, self.axis_dim, self.eps = embed_dim, axis_dim, eps
        self.rbf_proj_layer1 = ChannelLinear(n_radial, rbf_out_dim, backend=backend, bias=bias)
        self.rbf_proj_activation = activation(activation_function)
        self.rbf_proj_layer2 = ChannelLinear(rbf_out_dim, rbf_out_dim, backend=backend, bias=bias)
        self.env_type_embed = TypeEmbedding(env_type_table)
        self.g_layer1 = ChannelLinear(rbf_out_dim + 2 * type_dim, hidden_dim, backend=backend, bias=bias)
        self.g_activation = activation(activation_function)
        self.g_layer2 = ChannelLinear(hidden_dim, embed_dim, backend=backend, bias=bias)
        self.output_proj = ChannelLinear(embed_dim * axis_dim, 2 * channels, backend=backend)

    def forward(self, **inputs):
        edge_index = inputs['edge_index']
        src, dst = edge_index
        edge_vec, env = inputs['edge_vec'], inputs['env']
        rbf, atype = inputs['rbf'], inputs['atype']
        inv_sqrt, mask = inputs['inv_sqrt'], inputs['mask']
        n_nodes = inputs['type_feat'].shape[0]

        inv_r = torch.rsqrt(edge_vec.square().sum(-1, keepdim=True) + self.eps ** 2)
        scale = env * inv_r
        r_tilde = torch.cat((scale, scale * edge_vec * inv_r), dim=-1)
        r_tilde = torch.where(mask[:, None], r_tilde, 0.)

        type_src = self.env_type_embed(atype[src])['y']
        type_dst = self.env_type_embed(atype[dst])['y']
        rbf_proj = self.rbf_proj_layer2(self.rbf_proj_activation(self.rbf_proj_layer1(rbf)))
        edge_features = torch.cat((rbf_proj, type_src, type_dst), dim=-1)
        g = self.g_layer2(self.g_activation(self.g_layer1(edge_features)))
        outer = torch.einsum('ei,ej->eij', r_tilde, g).reshape(rbf.shape[0], -1)
        aggregate = outer.new_zeros((n_nodes, outer.shape[1])).index_add(0, dst, outer)
        aggregate = aggregate.reshape(n_nodes, 4, self.embed_dim) * inv_sqrt
        d_matrix = torch.bmm(aggregate.transpose(1, 2), aggregate[:, :, :self.axis_dim])
        return {'y': self.output_proj(d_matrix.reshape(n_nodes, -1))}


class Film(nn.Module):
    """Apply checkpointed, normalized environment FiLM to scalar features."""

    def __init__(self, scale_norm, shift_norm, scale_strength_log, shift_strength_log, *, eps=1e-7, backend='reference'):
        super().__init__()
        if scale_norm is None or shift_norm is None:
            self.film_scale_norm = nn.Identity()
            self.film_shift_norm = nn.Identity()
        else:
            channels = scale_norm.numel()
            self.film_scale_norm = ChannelRMSNorm(channels, eps=eps, backend=backend)
            self.film_shift_norm = ChannelRMSNorm(channels, eps=eps, backend=backend)
            with torch.no_grad():
                self.film_scale_norm.adam_scale.copy_(scale_norm)
                self.film_shift_norm.adam_scale.copy_(shift_norm)
        self.film_scale_strength_log = nn.Parameter(scale_strength_log.detach().clone())
        self.film_shift_strength_log = nn.Parameter(shift_strength_log.detach().clone())

    def forward(self, x, film):
        channels = x.shape[-1]
        scale_logits, shift_logits = film.split(channels, dim=-1)
        scale = 1.0 + torch.exp(self.film_scale_strength_log) * torch.tanh(self.film_scale_norm(scale_logits))
        shift = torch.exp(self.film_shift_strength_log) * torch.tanh(self.film_shift_norm(shift_logits))
        return {'y': x * scale + shift}


class GeometricSeed(nn.Module):
    """Zonal Wigner coupling of radial features into non-scalar node rows."""

    def __init__(self, lmax, channels, non_scalar_row_index, zonal_m0_col_index_for_row, radial_slot_index_for_row):
        super().__init__()
        dim = (int(lmax) + 1) ** 2
        n_rows = dim - 1
        if any(index.shape != (n_rows,) for index in (
            non_scalar_row_index, zonal_m0_col_index_for_row, radial_slot_index_for_row
        )):
            raise ValueError('GIE index buffers do not match lmax')
        if not torch.equal(non_scalar_row_index.cpu(), torch.arange(1, dim)):
            raise ValueError('GIE rows must follow packed non-scalar coefficient order')
        self.lmax, self.channels = int(lmax), int(channels)
        self.register_buffer('non_scalar_row_index', non_scalar_row_index.detach().clone())
        self.register_buffer('zonal_m0_col_index_for_row', zonal_m0_col_index_for_row.detach().clone())
        self.register_buffer('radial_slot_index_for_row', radial_slot_index_for_row.detach().clone())

    def forward(self, scalar, radial, Dt, edge_index, inv_sqrt, mask, env=None):
        n_nodes = scalar.shape[0]
        if self.lmax == 0:
            return {'y': scalar[:, None, None, :]}
        src, dst = edge_index
        zonal = Dt[:, self.non_scalar_row_index, self.zonal_m0_col_index_for_row]
        radial_non_scalar = radial[:, 1:, :].index_select(1, self.radial_slot_index_for_row)
        messages = zonal.unsqueeze(-1) * radial_non_scalar
        messages = torch.where(mask[:, None, None], messages, 0.)
        aggregate = messages.new_zeros((n_nodes, self.lmax * (self.lmax + 2), self.channels))
        aggregate = aggregate.index_add(0, dst, messages) * inv_sqrt
        features = torch.cat((scalar.unsqueeze(1), aggregate), dim=1)
        return {'y': features.unsqueeze(2)}


class DegreeLinear(nn.Module):
    """Checkpoint-layout SO(3) linear map shared across each degree's m rows."""

    def __init__(self, weight, expand_index, bias=None, *, output_singleton_focus=False):
        super().__init__()
        if weight.ndim != 3 or expand_index.ndim != 1:
            raise ValueError('SO(3) linear weights/index must have ranks 3 and 1')
        if expand_index.numel() == 0 or int(expand_index.max()) >= weight.shape[0]:
            raise ValueError('SO(3) linear degree index is outside its weight table')
        self.weight = nn.Parameter(weight.detach().clone())
        self.register_buffer('expand_index', expand_index.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.output_singleton_focus = bool(output_singleton_focus)

    def forward(self, x):
        if x.ndim == 4:
            if x.shape[2] != 1:
                raise ValueError('SO(3) linear expects a singleton focus axis')
            x = x.squeeze(2)
        if x.ndim != 3 or x.shape[1] != self.expand_index.numel():
            raise ValueError('SO(3) linear input does not match the packed degree index')
        weight = self.weight.index_select(0, self.expand_index)
        y = torch.einsum('ndc,dco->ndo', x, weight)
        if self.bias is not None:
            y = torch.cat((y[:, :1, :] + self.bias, y[:, 1:, :]), dim=1)
        if self.output_singleton_focus:
            y = y.unsqueeze(2)
        return {'y': y}


class GatherRotate(nn.Module):
    """Gather source nodes, rotate to the edge frame and retain m orders."""

    def __init__(self, lmax, mmax, n_focus, focus_channels):
        super().__init__()
        if not 0 <= mmax <= lmax:
            raise ValueError('SO(2) order must satisfy 0 <= mmax <= lmax')
        indices = [degree * degree + degree for degree in range(lmax + 1)]
        for order in range(1, mmax + 1):
            indices.extend(degree * degree + degree - order for degree in range(order, lmax + 1))
            indices.extend(degree * degree + degree + order for degree in range(order, lmax + 1))
        self.lmax, self.mmax = int(lmax), int(mmax)
        self.n_focus, self.focus_channels = int(n_focus), int(focus_channels)
        self.register_buffer('coefficient_index', torch.tensor(indices, dtype=torch.int64), persistent=False)

    def forward(self, x, edge_index, D, mask):
        dim = (self.lmax + 1) ** 2
        if x.ndim != 3 or x.shape[1] < dim or x.shape[2] != self.n_focus * self.focus_channels:
            raise ValueError('gather/rotate input does not match the block layout')
        if D.ndim != 3 or D.shape[1] < dim or D.shape[2] < dim:
            raise ValueError('Wigner matrix does not cover the block degree')
        src = edge_index[0]
        d_m = D[:, :dim, :dim].index_select(1, self.coefficient_index)
        local = torch.bmm(d_m, x.index_select(0, src)[:, :dim, :])
        local = local.reshape(local.shape[0], local.shape[1], self.n_focus, self.focus_channels)
        local = local.permute(0, 2, 1, 3)
        local = torch.where(mask[:, None, None, None], local, 0.)
        return {'y': local}


class _WeightLinear(nn.Module):
    """Linear projection preserving DeePMD's ``weight`` checkpoint spelling."""

    def __init__(self, weight):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError('radial projection weight must be rank two')
        self.weight = nn.Parameter(weight.detach().clone())

    def forward(self, x):
        return x @ self.weight


class _FocusGateLinear(nn.Module):
    def __init__(self, weight, n_focus, out_channels, bias=None):
        super().__init__()
        self.n_focus, self.out_channels = int(n_focus), int(out_channels)
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x):
        weight = self.weight.view(x.shape[-1], self.n_focus, self.out_channels)
        y = torch.einsum('bfi,ifo->bfo', x, weight)
        if self.bias is not None:
            y = y + self.bias.view(self.n_focus, self.out_channels)
        return y


class GatedActivation(nn.Module):
    """Scalar-activated l=0 and per-degree scalar gates for fndc tensors."""

    def __init__(self, lmax, mmax, channels, n_focus, expand_index, gate_weight,
                 activation_function='silu', gate_bias=None):
        super().__init__()
        self.lmax, self.mmax = int(lmax), int(mmax)
        self.channels, self.n_focus = int(channels), int(n_focus)
        expected = sum(2 * min(degree, self.mmax) + 1 for degree in range(self.lmax + 1)) - 1
        if expand_index.shape != (expected,):
            raise ValueError('gated activation index does not match its reduced layout')
        gate_out = self.lmax * self.channels
        if gate_weight.shape != (self.channels, self.n_focus * gate_out):
            raise ValueError('gated activation projection has an incompatible shape')
        self.register_buffer('expand_index', expand_index.detach().clone())
        self.gate_linear = _FocusGateLinear(gate_weight, self.n_focus, gate_out, gate_bias)
        self.scalar_activation = activation(activation_function)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.n_focus or x.shape[2] != self.expand_index.numel() + 1 or x.shape[3] != self.channels:
            raise ValueError('gated activation expects [E,F,Dm,C]')
        focus_major = x.permute(1, 0, 2, 3)
        scalar = focus_major[:, :, 0, :]
        logits = self.gate_linear(scalar.transpose(0, 1)).transpose(0, 1)
        logits = logits.reshape(self.n_focus, x.shape[0], self.lmax, self.channels)
        gates = torch.sigmoid(logits).index_select(2, self.expand_index)
        scalar_out = self.scalar_activation(focus_major[:, :, :1, :])
        nonscalar_out = focus_major[:, :, 1:, :] * gates
        return {'y': torch.cat((scalar_out, nonscalar_out), dim=2).permute(1, 0, 2, 3)}


class RadialSO2(nn.Module):
    """Checkpoint-shaped degree/channel radial mixer for the SO(2) path."""

    def __init__(self, lmax, mmax, channels, n_focus, focus_channels, mode, rank,
                 mixer_weight, kernel_compact_index, kernel_dense_index,
                 channel_basis=None, hidden_projection_weight=None):
        super().__init__()
        self.lmax, self.mmax = int(lmax), int(mmax)
        self.channels = int(channels)
        self.hidden_channels = int(n_focus * focus_channels)
        self.mode, self.rank = str(mode), int(rank)
        self.reduced_dim = (self.lmax + 1) + sum(
            2 * (self.lmax - order + 1) for order in range(1, self.mmax + 1)
        )
        self.degree_kernel_size = sum(
            (self.lmax - order + 1) ** 2 for order in range(self.mmax + 1)
        )
        degree_indices = list(range(self.lmax + 1))
        for order in range(1, self.mmax + 1):
            degree_indices.extend(range(order, self.lmax + 1))
            degree_indices.extend(range(order, self.lmax + 1))
        self.register_buffer('degree_index_m', torch.tensor(degree_indices, dtype=torch.int64), persistent=False)
        if hidden_projection_weight is None:
            self.radial_hidden_proj = None
            projected_channels = self.channels
        else:
            self.radial_hidden_proj = _WeightLinear(hidden_projection_weight)
            projected_channels = hidden_projection_weight.shape[1]
        if projected_channels != self.hidden_channels:
            raise ValueError('radial projection width does not match SO(2) hidden channels')

        if mode not in ('degree', 'degree_channel'):
            raise ValueError('radial SO(2) mode must be degree or degree_channel')
        if rank < 0:
            raise ValueError('radial SO(2) rank must be nonnegative')
        if mode == 'degree':
            expected_out = self.degree_kernel_size
        elif rank > 0:
            expected_out = self.degree_kernel_size * rank
        else:
            expected_out = self.degree_kernel_size * self.hidden_channels
        if mixer_weight.shape != ((self.lmax + 1) * self.hidden_channels, expected_out):
            raise ValueError('radial degree mixer weight shape does not match its config')
        if kernel_compact_index.shape != (self.degree_kernel_size,) or kernel_dense_index.shape != (self.degree_kernel_size,):
            raise ValueError('radial degree mixer scatter indices have invalid shapes')
        self.radial_degree_mixer = nn.Module()
        self.radial_degree_mixer.weight = nn.Parameter(mixer_weight.detach().clone())
        self.radial_degree_mixer.register_buffer('kernel_compact_index', kernel_compact_index.detach().clone())
        self.radial_degree_mixer.register_buffer('kernel_dense_index', kernel_dense_index.detach().clone())
        if channel_basis is not None:
            if channel_basis.shape != (rank, self.hidden_channels):
                raise ValueError('radial degree mixer channel basis has invalid shape')
            self.radial_degree_mixer.channel_basis = nn.Parameter(channel_basis.detach().clone())
        else:
            self.radial_degree_mixer.register_parameter('channel_basis', None)

    def _mix(self, x_local, radial_feat):
        n_edges = x_local.shape[0]
        kernel_flat = radial_feat[:, :self.lmax + 1, :].reshape(n_edges, -1) @ self.radial_degree_mixer.weight
        compact_index = self.radial_degree_mixer.kernel_compact_index
        dense_index = self.radial_degree_mixer.kernel_dense_index
        if self.mode == 'degree':
            dense = kernel_flat.new_zeros((n_edges, self.reduced_dim * self.reduced_dim))
            dense.index_copy_(1, dense_index, kernel_flat.index_select(1, compact_index))
            kernel = dense.view(n_edges, self.reduced_dim, self.reduced_dim)
            return torch.bmm(kernel, x_local)
        if self.rank > 0:
            compact = kernel_flat.view(n_edges, self.degree_kernel_size, self.rank)
            dense = compact.new_zeros((n_edges, self.reduced_dim * self.reduced_dim, self.rank))
            dense.index_copy_(1, dense_index, compact.index_select(1, compact_index))
            kernel = dense.view(n_edges, self.reduced_dim, self.reduced_dim, self.rank)
            mixed = torch.einsum('eoir,eic->eorc', kernel, x_local)
            basis = self.radial_degree_mixer.channel_basis.view(1, 1, self.rank, self.hidden_channels)
            return (mixed * basis).sum(dim=2)
        compact = kernel_flat.view(n_edges, self.degree_kernel_size, self.hidden_channels)
        dense = compact.new_zeros((n_edges, self.reduced_dim * self.reduced_dim, self.hidden_channels))
        dense.index_copy_(1, dense_index, compact.index_select(1, compact_index))
        kernel = dense.view(n_edges, self.reduced_dim, self.reduced_dim, self.hidden_channels)
        return torch.einsum('eoic,eic->eoc', kernel, x_local)

    def forward(self, x, radial):
        if x.ndim != 4 or x.shape[1:] != (self.n_focus, self.reduced_dim, self.hidden_channels):
            raise ValueError('radial SO(2) input does not match its focus/reduced layout')
        radial_projected = self.radial_hidden_proj(radial) if self.radial_hidden_proj is not None else radial
        radial_feat = radial_projected.index_select(1, self.degree_index_m)
        x_local = x.permute(0, 2, 1, 3).reshape(x.shape[0], self.reduced_dim, self.hidden_channels)
        mixed = self._mix(x_local, radial_feat)
        y = mixed.reshape(x.shape[0], self.reduced_dim, self.n_focus, self.hidden_channels)
        return {'y': y.permute(0, 2, 1, 3), 'radial': radial_projected}


class SO2Linear(nn.Module):
    """Real SO(2)-equivariant channel mixing in m-major local coordinates."""

    def __init__(self, lmax, mmax, n_focus, in_channels, weight_m0, weight_m,
                 m0_idx, pos_indices, neg_indices, bias0=None):
        super().__init__()
        self.lmax, self.mmax = int(lmax), int(mmax)
        self.n_focus, self.in_channels = int(n_focus), int(in_channels)
        num_l0 = self.lmax + 1
        if weight_m0.shape[0] != num_l0 * self.in_channels or weight_m0.shape[1] % (self.n_focus * num_l0):
            raise ValueError('SO(2) m=0 weight has incompatible shape')
        self.out_channels = weight_m0.shape[1] // (self.n_focus * num_l0)
        if m0_idx.shape != (num_l0,) or len(weight_m) != self.mmax:
            raise ValueError('SO(2) index/weight list does not match its config')
        self.weight_m0 = nn.Parameter(weight_m0.detach().clone())
        self.weight_m = nn.ParameterList([nn.Parameter(weight.detach().clone()) for weight in weight_m])
        self.register_buffer('m0_idx', m0_idx.detach().clone())
        self.register_buffer('pos_indices', pos_indices.detach().clone())
        self.register_buffer('neg_indices', neg_indices.detach().clone())
        self.bias0 = nn.Parameter(bias0.detach().clone()) if bias0 is not None else None

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.n_focus or x.shape[3] != self.in_channels:
            raise ValueError('SO(2) linear input must have shape [E,F,Dm,Cin]')
        focus_major = x.permute(1, 0, 2, 3)
        num_l0 = self.lmax + 1
        x_m0 = focus_major[:, :, :num_l0, :].reshape(self.n_focus, x.shape[0], -1)
        weight_m0 = self.weight_m0.view(
            num_l0 * self.in_channels, self.n_focus, num_l0 * self.out_channels
        )
        y_m0 = torch.einsum('fei,ifo->feo', x_m0, weight_m0).reshape(
            self.n_focus, x.shape[0], num_l0, self.out_channels
        )
        if self.bias0 is not None:
            bias = self.bias0.view(self.n_focus, self.out_channels)
            y_m0 = torch.cat((y_m0[:, :, :1, :] + bias[:, None, None, :], y_m0[:, :, 1:, :]), dim=2)
        pieces = [y_m0]
        index_offset = 0
        for order, weight in enumerate(self.weight_m, start=1):
            num_l = self.lmax - order + 1
            neg_idx = self.neg_indices[index_offset:index_offset + num_l]
            pos_idx = self.pos_indices[index_offset:index_offset + num_l]
            index_offset += num_l
            x_neg = focus_major.index_select(2, neg_idx).reshape(self.n_focus, x.shape[0], -1)
            x_pos = focus_major.index_select(2, pos_idx).reshape(self.n_focus, x.shape[0], -1)
            out_width = num_l * self.out_channels
            matrix = weight.view(num_l * self.in_channels, self.n_focus, 2 * out_width)
            w_u, w_v = matrix[:, :, :out_width], matrix[:, :, out_width:]
            xw = lambda value, w: torch.einsum('fei,ifo->feo', value, w).reshape(
                self.n_focus, x.shape[0], num_l, self.out_channels
            )
            y_neg = xw(x_neg, w_u) - xw(x_pos, w_v)
            y_pos = xw(x_neg, w_v) + xw(x_pos, w_u)
            pieces.extend((y_neg, y_pos))
        y = torch.cat(pieces, dim=2).permute(1, 0, 2, 3)
        return {'y': y}


class EquivariantNorm(nn.Module):
    """Degree-balanced RMS normalization for packed SO(3) coefficients."""

    def __init__(self, adam_scale, bias, expand_index, balance_weight, *, eps=1e-7):
        super().__init__()
        if adam_scale.ndim != 3 or bias.ndim != 2:
            raise ValueError('equivariant norm affine tensors have invalid ranks')
        if expand_index.ndim != 1 or balance_weight.shape != expand_index.shape:
            raise ValueError('equivariant norm degree tables have incompatible shapes')
        if adam_scale.shape[0] != int(expand_index.max()) + 1 or adam_scale.shape[1:] != bias.shape:
            raise ValueError('equivariant norm scale/bias dimensions disagree')
        self.adam_scale = nn.Parameter(adam_scale.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone())
        self.register_buffer('expand_index', expand_index.detach().clone())
        self.register_buffer('balance_weight', balance_weight.detach().clone())
        self.eps = float(eps)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.expand_index.numel() or x.shape[2:] != self.bias.shape:
            raise ValueError('equivariant norm input must be [N,D,F,C]')
        input_dtype = x.dtype
        x = x.to(self.adam_scale.dtype)
        x0 = x[:, :1] - x[:, :1].mean(dim=-1, keepdim=True)
        xt = x[:, 1:]
        variance = x0.square().sum(dim=(1, 3)) * self.balance_weight[0]
        if xt.numel():
            variance = variance + torch.einsum('ndfc,d->nf', xt.square(), self.balance_weight[1:])
        inv_rms = torch.rsqrt(variance + self.eps).unsqueeze(1).unsqueeze(-1)
        scale = self.adam_scale.index_select(0, self.expand_index).unsqueeze(0)
        x0 = x0 * inv_rms * scale[:, :1]
        if xt.numel():
            xt = xt * inv_rms * scale[:, 1:]
        x0 = x0 + self.bias.reshape(1, 1, *self.bias.shape)
        y = x0 if not xt.numel() else torch.cat((x0, xt), dim=1)
        return {'y': y.to(input_dtype)}


class InverseRotate(nn.Module):
    """Lift the reduced m-major local message back to the packed node basis."""

    def __init__(self, lmax, mmax, n_focus, focus_channels):
        super().__init__()
        self.lmax, self.mmax = int(lmax), int(mmax)
        self.n_focus, self.focus_channels = int(n_focus), int(focus_channels)
        coeff_index = [degree * degree + degree for degree in range(self.lmax + 1)]
        degree_index = list(range(self.lmax + 1))
        for order in range(1, self.mmax + 1):
            coeff_index.extend(degree * degree + degree - order for degree in range(order, self.lmax + 1))
            coeff_index.extend(degree * degree + degree + order for degree in range(order, self.lmax + 1))
            degree_index.extend(range(order, self.lmax + 1))
            degree_index.extend(range(order, self.lmax + 1))
        degree_tensor = torch.tensor(degree_index, dtype=torch.int64)
        scale = torch.ones(len(degree_index), dtype=torch.float32)
        if self.mmax < self.lmax:
            selected = degree_tensor > self.mmax
            scale[selected] = torch.sqrt(
                (2.0 * degree_tensor[selected].float() + 1.0) / float(2 * self.mmax + 1)
            )
        self.register_buffer('coefficient_index', torch.tensor(coeff_index, dtype=torch.int64), persistent=False)
        self.register_buffer('rotate_rescale', scale, persistent=False)

    def forward(self, x, Dt):
        expected_dim = sum(2 * min(degree, self.mmax) + 1 for degree in range(self.lmax + 1))
        if x.ndim != 4 or x.shape[1:] != (self.n_focus, expected_dim, self.focus_channels):
            raise ValueError('inverse rotation input does not match the block layout')
        full_dim = (self.lmax + 1) ** 2
        if Dt.ndim != 3 or Dt.shape[1] < full_dim or Dt.shape[2] < full_dim:
            raise ValueError('inverse rotation matrix does not cover the block degree')
        local = x.permute(0, 2, 1, 3).reshape(x.shape[0], expected_dim, -1)
        dt_from_m = Dt[:, :full_dim, :full_dim].index_select(2, self.coefficient_index)
        message = torch.bmm(dt_from_m, local)
        return {'y': message * self.rotate_rescale.to(message.dtype).view(1, -1, 1)}


class _FocusRMSNorm(nn.Module):
    def __init__(self, scale, eps):
        super().__init__()
        self.adam_scale = nn.Parameter(scale.detach().clone())
        self.eps = float(eps)

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.adam_scale.unsqueeze(0)


class _FocusLinear(nn.Module):
    def __init__(self, weight, n_focus, out_channels):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.n_focus, self.out_channels = int(n_focus), int(out_channels)

    def forward(self, x):
        weight = self.weight.view(x.shape[-1], self.n_focus, self.out_channels)
        return torch.einsum('nfi,ifo->nfo', x, weight)


class AttentionLogits(nn.Module):
    """Per-edge Q/K logits and per-node output gates for SeZM attention."""

    def __init__(self, focuses, channels, heads, state, *, eps=1e-7):
        super().__init__()
        if channels % heads:
            raise ValueError('attention heads must divide the focus width')
        self.focuses, self.channels, self.heads = int(focuses), int(channels), int(heads)
        self.head_dim = self.channels // self.heads
        self.attn_qk_norm = _FocusRMSNorm(state['attn_qk_norm.adam_scale'], eps)
        self.attn_q_proj = _FocusLinear(state['attn_q_proj.weight'], focuses, channels)
        self.attn_k_proj = _FocusLinear(state['attn_k_proj.weight'], focuses, channels)
        self.attn_output_gate_norm = _FocusRMSNorm(state['attn_output_gate_norm.adam_scale'], eps)
        self.adamw_attn_logit_w = nn.Parameter(state['adamw_attn_logit_w'].detach().clone())
        self.adamw_attn_gate_w = nn.Parameter(state['adamw_attn_gate_w'].detach().clone())

    def forward(self, x, radial, edge_index):
        x_l0 = x[:, 0, :].reshape(x.shape[0], self.focuses, self.channels)
        normalized = self.attn_qk_norm(x_l0)
        q_node, k_node = self.attn_q_proj(normalized), self.attn_k_proj(normalized)
        src, dst = edge_index
        n_edges = src.numel()
        q_edge = q_node.index_select(0, dst).reshape(n_edges, self.focuses, self.heads, self.head_dim)
        k_edge = k_node.index_select(0, src).reshape(n_edges, self.focuses, self.heads, self.head_dim)
        logits = (q_edge * k_edge).sum(-1) * (self.head_dim ** -0.5)
        radial_l0 = radial[:, 0, :].reshape(n_edges, self.focuses, self.channels)
        logits = logits + torch.einsum('efi,ifo->efo', radial_l0, self.adamw_attn_logit_w)
        gate_logits = torch.einsum(
            'nfi,ifo->nfo',
            self.attn_output_gate_norm(x_l0),
            self.adamw_attn_gate_w,
        )
        return {'logits': logits, 'gate': torch.sigmoid(gate_logits)}


class FocusCompetition(nn.Module):
    """Smoothly reweight focus streams using their pre-mixing scalar features."""

    def __init__(self, weight, n_focus, channels, *, norm_scale=None, bias=None, eps=1e-7,
                 temperature=1.0, label_smoothing=0.02):
        super().__init__()
        if weight.shape != (channels, n_focus):
            raise ValueError('focus competition weight must have shape [C,F]')
        self.n_focus, self.channels = int(n_focus), int(channels)
        self.temperature, self.label_smoothing = float(temperature), float(label_smoothing)
        self.focus_compete_norm = (
            _FocusRMSNorm(norm_scale, eps) if norm_scale is not None else nn.Identity()
        )
        self.adamw_focus_compete_w = nn.Parameter(weight.detach().clone())
        self.focus_compete_bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x, gate):
        if x.ndim != 4 or x.shape[1] != self.n_focus or x.shape[3] != self.channels:
            raise ValueError('focus competition expects local values [E,F,M,C]')
        if gate.shape != (x.shape[0], self.n_focus, self.channels):
            raise ValueError('focus competition gate must have shape [E,F,C]')
        logits = torch.einsum(
            'efi,if->ef', self.focus_compete_norm(gate), self.adamw_focus_compete_w
        )
        if self.focus_compete_bias is not None:
            logits = logits + self.focus_compete_bias
        alpha = torch.softmax(logits / self.temperature, dim=1)
        alpha = alpha * (1.0 - self.label_smoothing) + self.label_smoothing / self.n_focus
        return {'y': x * alpha[:, :, None, None]}


class AttentionSoftmax(nn.Module):
    """Destination-segmented, envelope-gated softmax with physical null mass."""

    def __init__(self, z_bias_raw, *, eps=1e-7):
        super().__init__()
        self.adamw_attn_z_bias_raw = nn.Parameter(z_bias_raw.detach().clone())
        self.eps = float(eps)

    def forward(self, logits, env, edge_index, degree, mask):
        if logits.ndim != 3 or env.shape != (logits.shape[0], 1):
            raise ValueError('attention softmax expects logits [E,F,H] and env [E,1]')
        n_edges, n_focus, n_heads = logits.shape
        n_channels = n_focus * n_heads
        input_dtype = logits.dtype
        compute_dtype = torch.float32 if input_dtype in (torch.float16, torch.bfloat16) else input_dtype
        logits_flat = logits.reshape(n_edges, n_channels).to(compute_dtype)
        edge_env = env.reshape(n_edges).to(compute_dtype)
        active = (edge_env > 0) & mask
        safe_env = torch.where(active, edge_env, torch.ones_like(edge_env))
        effective = logits_flat + (2.0 * torch.log(safe_env)).unsqueeze(1)
        effective = torch.where(active[:, None], effective, torch.full_like(effective, float('-inf')))
        null_mass = (torch.nn.functional.softplus(self.adamw_attn_z_bias_raw.to(compute_dtype)) + self.eps).reshape(1, n_channels)
        null_logit = torch.log(null_mass)
        dst = edge_index[1]
        dst_index = dst.reshape(n_edges, 1).expand(n_edges, n_channels)
        n_nodes = degree.shape[0]
        group_max = null_logit.expand(n_nodes, n_channels).clone()
        group_max = torch.scatter_reduce(group_max, 0, dst_index, effective,
                                         reduce='amax', include_self=True)
        edge_max = group_max.index_select(0, dst)
        edge_exp = torch.exp(effective - edge_max)
        denom_sum = torch.zeros(n_nodes, n_channels, dtype=compute_dtype, device=logits.device)
        denom_sum = torch.scatter_add(denom_sum, 0, dst_index, edge_exp)
        denominator = denom_sum + torch.exp(null_logit - group_max)
        alpha = edge_exp / denominator.index_select(0, dst)
        return {'y': alpha.to(input_dtype).reshape(n_edges, n_focus, n_heads)}


class _GLUFittingLayer(nn.Module):
    def __init__(self, matrix, bias, activation_function, backend):
        super().__init__()
        if matrix.shape[1] % 2:
            raise ValueError('GLU fitting matrix output width must be even')
        hidden = matrix.shape[1] // 2
        if bias is None or bias.shape != (matrix.shape[1],):
            raise ValueError('GLU fitting hidden bias has an incompatible shape')
        self.linear = ChannelLinear(matrix.shape[0], matrix.shape[1], backend=backend, bias=True)
        with torch.no_grad():
            self.linear.matrix.copy_(matrix)
            self.linear.bias.copy_(bias)
        self.activate = activation(activation_function)
        self.hidden = hidden

    def forward(self, x):
        value, gate = self.linear(x).split(self.hidden, dim=-1)
        return value * self.activate(gate)


class FittingEnergy(nn.Module):
    """Single-task SeZM GLU fitting network and per-type energy offsets."""

    def __init__(self, hidden_layers, output_matrix, output_bias, bias_atom_e, out_bias,
                 *, activation_function='silu', backend='reference'):
        super().__init__()
        if out_bias.ndim == 3:
            if out_bias.shape[0] != 1:
                raise ValueError('single-task energy requires one output-stat table')
            out_bias = out_bias[0]
        if bias_atom_e.ndim == 1:
            bias_atom_e = bias_atom_e[:, None]
        if out_bias.ndim == 1:
            out_bias = out_bias[:, None]
        if bias_atom_e.ndim != 2 or bias_atom_e.shape[1] != 1 or out_bias.shape != bias_atom_e.shape:
            raise ValueError('per-type energy offsets must have shape [T,1]')
        network = nn.Module()
        network.hidden_layers = nn.ModuleList([
            _GLUFittingLayer(matrix, bias, activation_function, backend)
            for matrix, bias in hidden_layers
        ])
        if output_matrix.ndim != 2 or output_matrix.shape[1] != 1:
            raise ValueError('energy fitting output matrix must have shape [H,1]')
        network.output_layer = ChannelLinear(
            output_matrix.shape[0], 1, backend=backend, bias=output_bias is not None
        )
        with torch.no_grad():
            network.output_layer.matrix.copy_(output_matrix)
            if output_bias is not None:
                network.output_layer.bias.copy_(output_bias.reshape(1))
        self.filter_layers = nn.Module()
        self.filter_layers.networks = nn.ModuleList([network])
        self.register_buffer('bias_atom_e', bias_atom_e.detach().clone())
        self.register_buffer('out_bias', out_bias.detach().clone())

    def forward(self, descriptor, atype):
        x = descriptor
        network = self.filter_layers.networks[0]
        for layer in network.hidden_layers:
            x = layer(x)
        atom_energy = network.output_layer(x)
        atom_energy = atom_energy + self.bias_atom_e[atype] + self.out_bias[atype]
        return {'y': atom_energy.to(torch.float64)}


class ScalarReadout(nn.Module):
    """Residual scalar readout around a configured SO(3) grid FFN."""

    def __init__(self, output_ffn):
        super().__init__()
        self.output_ffn = output_ffn

    def forward(self, x):
        if x.ndim != 4 or x.shape[2] != 1:
            raise ValueError('scalar readout expects node features [N,D,1,C]')
        residual = x[:, 0, 0, :]
        update = self.output_ffn.forward_scalar(x)[:, 0, 0, :]
        return {'y': residual + update}


class TensorResult(nn.Module):
    """Adapt a tensor-returning composition to the backend's named-output API."""

    def __init__(self, module, input_names=('x',)):
        super().__init__()
        self.module = module
        self.input_names = tuple(input_names)

    def forward(self, **inputs):
        return {'y': self.module(*(inputs[name] for name in self.input_names))}


class RadialBasis(nn.Module):
    def __init__(self, config, frequencies, *, backend='reference'):
        super().__init__()
        if frequencies.shape != (1, config.n_radial) or not frequencies.is_floating_point():
            raise ValueError('radial frequencies must be floating [1,R]')
        self.config, self.backend = config, backend
        self.adam_freqs = nn.Parameter(frequencies.detach().clone())

    def forward(self, distance, mask):
        rbf, env = ops.radial(distance, self.adam_freqs, self.config.rcut,
                              *self.config.env_exp, backend=self.backend)
        return {'rbf': torch.where(mask[:, None], rbf, 0.),
                'env': torch.where(mask[:, None], env, 0.)}
