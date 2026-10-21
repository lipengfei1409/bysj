"""DPA4 composition skeleton with injectable operators and a full shape path.

Numerical execution intentionally raises at unfinished operators. The shape
backend traces conservative energy/force plumbing without fabricating values.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import torch
from torch import nn

from .backend import PlaceholderBackend, TensorSpec
from .config import DPA4Config
from .types import AtomicBatch, EdgeGraph, EdgeFeatures, ModelOutput
from .numerical import NumericalBackend


class DPA4InteractionBlock(nn.Module):
    def __init__(self, config, index, backend):
        super().__init__()
        self.config, self.index = config, index
        # The parent DPA4Model owns this backend once; blocks only borrow it.
        self.__dict__['backend'] = backend

    def forward(self, x, radial, cache, graph):
        c = self.config
        unsupported = []
        if c.optional.get('mlp_bias', False):
            unsupported.append('mlp_bias')
        if c.optional.get('layer_scale', False):
            unsupported.append('layer_scale')
        for flag in ('edge_cartesian', 'node_wise_s2', 'node_wise_so3', 'message_node_s2'):
            if c.optional.get(flag, False):
                unsupported.append(flag)
        for mode in ('so2_attn_res', 'full_attn_res', 'block_attn_res'):
            if c.optional.get(mode) not in (None, 'none'):
                unsupported.append(mode)
        if c.optional.get('node_cartesian') not in (None, 'none', 0, '0'):
            unsupported.append('node_cartesian')
        if unsupported:
            raise NotImplementedError(f'active block options are not implemented: {sorted(unsupported)}')
        if c.so2_norm:
            raise NotImplementedError("intermediate SO(2) equivariant norms are not wired in the independent model path")
        n, e = x.shape[0], graph.mask.shape[0]
        l, m = c.degrees[self.index], c.orders[self.index]
        dn = (l + c.extra_node_l + 1) ** 2
        dm = sum(2 * min(degree, m) + 1 for degree in range(l + 1))
        wide = c.n_focus * c.focus_channels
        tag = f"blocks.{self.index}"

        def op(name, inputs, shape, suffix="", dtype=None):
            dtype = x.dtype if dtype is None else dtype
            return self.backend.run(name, inputs, {"y": TensorSpec(shape, dtype)}, stage=f"{tag}.{suffix or name}")["y"]

        if x.shape[1] != dn:
            x = op("degree_select", {"x": x}, (n, dn, 1, c.channels))
        residual_input = x
        radial = radial[:, :l + 1, :]
        if c.sandwich_norm[0]:
            x = op("equivariant_norm", {"x": x}, (n, dn, 1, c.channels), "pre_so2_norm")
        focused = op("focus_mix", {"x": x}, (n, dn, wide))
        local = op("gather_rotate", {"x": focused, "edge_index": graph.edge_index, "D": cache["D"], "mask": graph.mask}, (e, c.n_focus, dm, c.focus_channels))
        modulation = "radial_modulation" if c.radial_so2_mode == "none" else "radial_so2"
        if modulation == "radial_so2":
            radial_result = self.backend.run(
                modulation,
                {"x": local, "radial": radial},
                {
                    "y": TensorSpec(tuple(local.shape), x.dtype),
                    "radial": TensorSpec((e, l + 1, c.n_focus * c.focus_channels), x.dtype),
                },
                stage=f"{tag}.{modulation}",
            )
            local, radial_attention = radial_result["y"], radial_result["radial"]
        else:
            local = op(modulation, {"x": local, "radial": radial}, tuple(local.shape))
            radial_attention = radial
        focus_gate = local[:, :, 0, :]
        for j in range(c.mixing_layers):
            residual = local
            local = op("so2_linear", {"x": local}, tuple(local.shape), f"so2_linear.{j}")
            if j < c.mixing_layers - 1:
                local = op("gated_activation", {"x": local}, tuple(local.shape), f"so2_activation.{j}")
            local = residual + local
        if c.focus_compete and c.n_focus > 1:
            local = op("focus_compete", {"x": local, "gate": focus_gate}, tuple(local.shape))
        message = op("inverse_rotate", {"x": local, "Dt": cache["Dt"]}, (e, (l + 1) ** 2, wide))
        if c.n_heads:
            fa = 1 if c.atten_f_mix else c.n_focus
            attention = self.backend.run("attention_logits", {"x": focused, "radial": radial_attention, "edge_index": graph.edge_index},
                {"logits": TensorSpec((e, fa, c.n_heads), x.dtype), "gate": TensorSpec((n, fa, c.n_heads), x.dtype)}, stage=f"{tag}.attention_logits")
            alpha = op("attention_softmax", {"logits": attention["logits"], "env": cache["env"], "edge_index": graph.edge_index,
                        "degree": cache['degree'], "mask": graph.mask}, (e, fa, c.n_heads))
            reduce_inputs = {"message": message, "alpha": alpha, "gate": attention["gate"], "edge_index": graph.edge_index}
        else:
            reduce_inputs = {"message": message, "env": cache["env"], "edge_index": graph.edge_index}
        reduce_inputs.update(inv_sqrt=cache['inv_sqrt'], mask=graph.mask)
        update = op("message_reduce", reduce_inputs, (n, (l + 1) ** 2, wide))
        if c.message_node_so3:
            grid_update = op("message_node_so3", {"query": update, "context": focused}, tuple(update.shape))
            update = update + grid_update
        update = op("focus_unmix", {"x": update}, (n, dn, 1, c.channels))
        if c.sandwich_norm[1]:
            update = op("equivariant_norm", {"x": update}, tuple(update.shape), "post_so2_norm")
        # The residual owns the original pre-normalization backbone.
        x = op("residual", {"x": residual_input, "update": update}, (n, dn, 1, c.channels))
        ffn_operator = "so3_grid_ffn" if c.ffn_so3_grid else "equivariant_ffn"
        for j in range(c.ffn_blocks):
            norm = op("equivariant_norm", {"x": x}, tuple(x.shape), f"pre_ffn_norm.{j}") if c.sandwich_norm[2] else x
            update = op(ffn_operator, {"x": norm}, tuple(x.shape), f"ffn.{j}")
            if c.sandwich_norm[3]:
                update = op("equivariant_norm", {"x": update}, tuple(x.shape), f"post_ffn_norm.{j}")
            x = op("residual", {"x": x, "update": update}, tuple(x.shape), f"ffn_residual.{j}")
        return x

class DPA4Model(nn.Module):
    def __init__(self, config: DPA4Config, backend=None):
        super().__init__()
        self.config = config
        self.backend = backend if backend is not None else PlaceholderBackend()
        self.blocks = nn.ModuleList([DPA4InteractionBlock(config, i, self.backend) for i in range(config.n_blocks)])

    @classmethod
    def from_checkpoint(cls, path, *, head=None, primitive_backend='reference', linear_backend='reference'):
        """Load supported checkpoint weights; inspect the completeness report."""
        from .loading import load_frontend_checkpoint
        return load_frontend_checkpoint(path, head=head, primitive_backend=primitive_backend, linear_backend=linear_backend)

    def plan_execution(self, *, n_atoms, edge_capacity, n_frames, edge_tile=256, node_tile=128, fuse_edge_features=None):
        """Return the explicit task/epoch plan; this does not launch kernels."""
        from dcu_dpa4.megakernel import build_dpa4_task_graph

        if fuse_edge_features is None:
            fuse_edge_features = (
                isinstance(self.backend, NumericalBackend)
                and self.backend.primitive_backend == 'hip'
            )
        return build_dpa4_task_graph(
            self.config,
            n_atoms=n_atoms,
            edge_capacity=edge_capacity,
            n_frames=n_frames,
            edge_tile=edge_tile,
            node_tile=node_tile,
            fuse_edge_features=fuse_edge_features,
        )

    def build_graph(self, batch: AtomicBatch, edge_capacity: int):
        batch.validate()
        if edge_capacity < 0:
            raise ValueError("edge_capacity must be nonnegative")
        tensors = self.backend.run("neighbor_graph", {"positions": batch.positions, "frame_index": batch.frame_index, "cells": batch.cells},
            {"edge_index": TensorSpec((2, edge_capacity), torch.int64), "shifts": TensorSpec((edge_capacity, 3), torch.int64),
             "frame_index": TensorSpec((edge_capacity,), torch.int64), "mask": TensorSpec((edge_capacity,), torch.bool)}, stage="neighbor_graph")
        return EdgeGraph(**tensors)

    def prepare_edges(self, batch: AtomicBatch, graph: EdgeGraph, *, fuse_edge_features=None) -> EdgeFeatures:
        """Run the learned numerical frontend, independently of unfinished blocks."""
        batch.validate()
        graph.validate()
        if batch.positions.device != graph.edge_index.device:
            raise ValueError("batch and graph must share a device")
        if isinstance(self.backend, NumericalBackend):
            self.backend.validate_inputs(batch, graph)
        c = self.config
        n, e, f = batch.positions.shape[0], graph.mask.shape[0], batch.n_frames
        dtype = getattr(self.backend, 'compute_dtype', torch.float32)

        def op(name, inputs, outputs):
            return self.backend.run(name, inputs, {key: TensorSpec(shape, dtype) for key, shape in outputs.items()}, stage=name)

        cells = batch.cells if batch.cells is not None else batch.positions.new_zeros(f, 3, 3)
        if fuse_edge_features is None:
            fuse_edge_features = (
                isinstance(self.backend, NumericalBackend)
                and self.backend.primitive_backend == 'hip'
            )
        if fuse_edge_features and not isinstance(self.backend, NumericalBackend):
            raise ValueError('fused edge features require a NumericalBackend')
        if fuse_edge_features:
            radial_module = (self.backend.operator_modules['radial_basis']
                             if 'radial_basis' in self.backend.operator_modules else None)
            if radial_module is None:
                raise RuntimeError('fused edge features require a registered radial basis')
            edge_feature_inputs = {
                "positions": batch.positions,
                "edge_index": graph.edge_index,
                "shifts": graph.shifts,
                "cells": cells,
                "edge_frame": graph.frame_index,
                "mask": graph.mask,
                "frequencies": radial_module.adam_freqs,
                "rcut": c.rcut,
                "basis_exponent": c.env_exp[0],
                "edge_exponent": c.env_exp[1],
                "eps": float(c.optional.get('eps', 1e-7)),
            }
            fused = op(
                "edge_features",
                edge_feature_inputs,
                {
                    "vec": (e, 3),
                    "distance": (e, 1),
                    "unit": (e, 3),
                    "rbf": (e, c.n_radial),
                    "env": (e, 1),
                },
            )
            geom = {key: fused[key] for key in ('vec', 'distance', 'unit')}
            radial_cache = {key: fused[key] for key in ('rbf', 'env')}
        else:
            geom = op("edge_geometry", {"positions": batch.positions, "edge_index": graph.edge_index, "shifts": graph.shifts, "cells": cells, "edge_frame": graph.frame_index, 'mask': graph.mask},
                      {"vec": (e, 3), "distance": (e, 1), "unit": (e, 3)})
            radial_cache = op("radial_basis", {"distance": geom["distance"], "mask": graph.mask},
                              {"rbf": (e, c.n_radial), "env": (e, 1)})
        type_feat = op("type_embedding", {"atype": batch.atype}, {"y": (n, c.channels)})["y"]
        if c.charge_spin_enabled:
            charge_spin = batch.charge_spin
            if charge_spin is None:
                if c.default_charge_spin is None:
                    raise ValueError("charge_spin is required by this checkpoint")
                charge_spin = batch.positions.new_tensor(c.default_charge_spin).reshape(1, 2)
            else:
                charge_spin = charge_spin.to(device=batch.positions.device, dtype=dtype)
                if charge_spin.ndim == 1:
                    charge_spin = charge_spin.reshape(1, 2)
            if charge_spin.shape[0] == 1 and f > 1:
                charge_spin = charge_spin.expand(f, -1)
            if charge_spin.shape != (f, 2):
                raise ValueError("charge_spin must have shape [2], [1,2] or [F,2]")
            condition = op("charge_spin_embedding", {"charge_spin": charge_spin}, {"y": (f, c.channels)})["y"]
            type_feat = type_feat + condition[batch.frame_index]
        pair = op("pair_embedding", {"x": type_feat, "edge_index": graph.edge_index, "mask": graph.mask}, {"y": (e, c.channels)})["y"]
        degree = op("degree_reduce", {"env": radial_cache["env"], "edge_index": graph.edge_index, "mask": graph.mask},
                    {"degree": (n,), "inv_sqrt": (n, 1, 1)})
        radial = op("radial_mlp", {"rbf": radial_cache["rbf"], "env": radial_cache["env"]}, {"y": (e, c.initial_l + 1, c.channels)})["y"]
        if isinstance(self.backend, NumericalBackend) and self.backend.version >= 1.1:
            radial = radial * radial_cache['env'][:, :, None]
        radial_fused = op("radial_fuse", {"radial": radial, "pair": pair}, {"y": tuple(radial.shape)})['y']
        return EdgeFeatures(**geom, **radial_cache, type_feat=type_feat, pair=pair, **degree,
                            radial=radial, radial_fused=radial_fused)

    def forward(self, batch: AtomicBatch, graph: EdgeGraph | None = None, *, edge_capacity: int | None = None,
                compute_forces=True, fuse_edge_features=None):
        numerical_forces = isinstance(self.backend, NumericalBackend) and compute_forces
        context = torch.enable_grad() if numerical_forces else nullcontext()
        with context:
            # Do not mutate the caller's requires_grad flag, even under no_grad.
            if numerical_forces and not batch.positions.requires_grad:
                batch = replace(batch, positions=batch.positions.detach().requires_grad_(True))
            return self._forward(
                batch, graph, edge_capacity=edge_capacity,
                compute_forces=compute_forces, fuse_edge_features=fuse_edge_features,
            )

    def _forward(self, batch, graph, *, edge_capacity, compute_forces, fuse_edge_features):
        if graph is None:
            if edge_capacity is None:
                raise ValueError("provide an EdgeGraph or edge_capacity for graph construction")
            graph = self.build_graph(batch, edge_capacity)
        front = self.prepare_edges(batch, graph, fuse_edge_features=fuse_edge_features)
        c = self.config
        n, e, f = batch.positions.shape[0], graph.mask.shape[0], batch.n_frames
        d0 = (c.initial_l + 1) ** 2
        dtype = front.radial.dtype

        def op(name, inputs, outputs):
            return self.backend.run(name, inputs, {key: TensorSpec(shape, out_dtype) for key, (shape, out_dtype) in outputs.items()}, stage=name)

        rotation = op("wigner_rotation", {"unit": front.unit}, {"D": ((e, d0, d0), dtype), "Dt": ((e, d0, d0), dtype)})
        scalar = front.type_feat
        if c.use_env_seed:
            film = op("environment_seed", {"rbf": front.rbf, "type_feat": front.type_feat, "pair": front.pair,
                "inv_sqrt": front.inv_sqrt, "edge_index": graph.edge_index, 'atype': batch.atype,
                'edge_vec': front.vec, 'env': front.env, 'mask': graph.mask}, {"y": ((n, 2*c.channels), dtype)})["y"]
            scalar = op("film", {"x": front.type_feat, "film": film}, {"y": ((n, c.channels), dtype)})["y"]
        x = op("geometric_seed", {"scalar": scalar, "radial": front.radial, "Dt": rotation["Dt"], "env": front.env,
            "edge_index": graph.edge_index, 'inv_sqrt': front.inv_sqrt, 'mask': graph.mask}, {"y": ((n, d0, 1, c.channels), dtype)})["y"]
        cache = {**rotation, 'rbf': front.rbf, 'env': front.env, 'degree': front.degree, 'inv_sqrt': front.inv_sqrt}
        for block in self.blocks:
            x = block(x, front.radial_fused, cache, graph)
        descriptor = op("scalar_readout", {"x": x}, {"y": ((n, c.channels), dtype)})["y"]
        atom_energy = op("fitting_energy", {"descriptor": descriptor, "atype": batch.atype}, {"y": ((n, 1), torch.float64)})["y"]
        energy = op("energy_reduce", {"atom_energy": atom_energy, "frame_index": batch.frame_index}, {"y": ((f, 1), torch.float64)})["y"]
        force = virial = None
        if compute_forces:
            gradient = op("edge_gradient", {"energy": energy, "edge_vec": front.vec}, {"y": ((e, 3), dtype)})["y"]
            derivatives = op("force_virial", {"gradient": gradient, "edge_vec": front.vec, "edge_index": graph.edge_index, "edge_frame": graph.frame_index, "mask": graph.mask},
                             {"force": ((n, 3), dtype), "virial": ((f, 3, 3), torch.float64)})
            force, virial = derivatives["force"], derivatives["virial"]
        return ModelOutput(energy, atom_energy, force, virial)
