"""Incremental numerical operator registry with strict tensor contracts."""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from dcu_dpa4.catalog import OPERATORS
from dcu_dpa4 import ops


class NumericalBackend(nn.Module):
    """One owner for learned operators; unfinished boundaries remain explicit.

    ``register(name, module, stage=...)`` uses ``module(**inputs) -> dict``.
    Stage-specific modules override a global operator. Built-ins implement
    only parameter-free primitives, never invented learned weights.
    """
    BUILTINS = frozenset(('edge_geometry', 'edge_features', 'pair_embedding', 'degree_reduce', 'radial_fuse',
                          'degree_select', 'message_reduce', 'residual', 'energy_reduce', 'edge_gradient', 'force_virial'))

    def __init__(self, config, *, primitive_backend='reference', dtype=torch.float32, version=1.0):
        super().__init__()
        if primitive_backend not in ('reference', 'hip'):
            raise ValueError('primitive backend must be reference or hip')
        if dtype not in (torch.float32, torch.float64):
            raise TypeError('modeling dtype must be float32 or float64')
        if not math.isfinite(version) or version < 1:
            raise ValueError('unsupported descriptor version')
        self.config, self.primitive_backend, self.version = config, primitive_backend, float(version)
        self.register_buffer('_anchor', torch.empty(0, dtype=dtype), persistent=False)
        self.operator_modules = nn.ModuleDict()
        self.stage_modules = nn.ModuleDict()
        self._stages = {}

    @property
    def compute_dtype(self):
        return self._anchor.dtype

    def get_extra_state(self):
        # Forward semantics cannot be inferred from learned tensor dimensions.
        return {'descriptor_version': self.version}

    def set_extra_state(self, state):
        version = float(state['descriptor_version'])
        if not math.isfinite(version) or version < 1:
            raise ValueError('unsupported serialized descriptor version')
        self.version = version

    def register(self, operator, module, *, stage=None):
        if operator not in OPERATORS:
            raise KeyError(operator)
        if not isinstance(module, nn.Module):
            raise TypeError('operator implementation must be nn.Module')
        key = operator if stage is None else (operator + ':' + stage).encode().hex()
        registry = self.operator_modules if stage is None else self.stage_modules
        if key in registry:
            raise ValueError(f'{operator} at {stage} already registered')
        registry[key] = module
        if stage is not None:
            self._stages[key] = (operator, stage)
        return self

    def report(self):
        result = [{'operator': name, 'stage': None, 'implemented': name in self.operator_modules or name in self.BUILTINS,
                   'implementation': 'module' if name in self.operator_modules else 'primitive' if name in self.BUILTINS else 'reserved'}
                  for name in OPERATORS]
        result.extend({'operator': name, 'stage': stage, 'implemented': True, 'implementation': 'module'}
                      for name, stage in self._stages.values())
        return result

    def validate_inputs(self, batch, graph):
        """Admission checks synchronize device scalars; not a performance path."""
        if batch.positions.dtype != self.compute_dtype or batch.positions.device != self._anchor.device:
            raise ValueError('positions must match model dtype/device; move model and batch explicitly')
        if batch.cells is not None and batch.cells.dtype != batch.positions.dtype:
            raise ValueError('cells and positions must share dtype')
        n = batch.positions.shape[0]
        def outside(t, upper):
            return bool(((t < 0) | (t >= upper)).any())
        if outside(batch.atype, len(self.config.type_map)) or outside(batch.frame_index, batch.n_frames):
            raise ValueError('atom types or frame indices out of range')
        if outside(graph.edge_index, n) or outside(graph.frame_index, batch.n_frames):
            raise ValueError('edge indices or frames out of range (including padding slots)')
        src, dst = graph.edge_index
        if bool(((batch.frame_index[src] != graph.frame_index) | (batch.frame_index[dst] != graph.frame_index)).any()):
            raise ValueError('cross-frame edge or incorrect edge frame')
        if batch.cells is None and bool((graph.shifts != 0).any()):
            raise ValueError('nonperiodic batches require zero shifts')
        if not bool(torch.isfinite(batch.positions).all()) or (batch.cells is not None and not bool(torch.isfinite(batch.cells).all())):
            raise ValueError('coordinates/cells must be finite')

    def run(self, operator, inputs, outputs, *, stage):
        if operator not in OPERATORS:
            raise KeyError(operator)
        stage_key = (operator + ':' + stage).encode().hex()
        if stage_key in self.stage_modules:
            result = self.stage_modules[stage_key](**inputs)
        elif operator in self.operator_modules:
            result = self.operator_modules[operator](**inputs)
        elif operator in self.BUILTINS:
            result = self._primitive(operator, inputs, outputs)
        else:
            raise NotImplementedError(f'operator {operator!r} at stage {stage!r} is not implemented; register an nn.Module')
        if not isinstance(result, Mapping) or set(result) != set(outputs):
            raise ValueError(f'{operator} at {stage}: output names must be {list(outputs)}')
        for name, spec in outputs.items():
            value = result[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != spec.shape:
                raise ValueError(f'{operator} at {stage}: {name} shape must be {spec.shape}')
            if value.dtype != spec.dtype:
                raise TypeError(f'{operator} at {stage}: {name} dtype must be {spec.dtype}, got {value.dtype}')
            if value.device != self._anchor.device:
                raise ValueError(f'{operator} at {stage}: {name} device must match model')
        return result

    def _primitive(self, name, x, outputs):
        native = self.primitive_backend
        if name == 'edge_features':
            vec, distance, unit, rbf, env = ops.edge_features(
                x['positions'], x['edge_index'], x['shifts'], x['cells'],
                x['edge_frame'], x['mask'], x['frequencies'], x['rcut'],
                x['basis_exponent'], x['edge_exponent'], x['eps'], backend=native,
            )
            return {'vec': vec, 'distance': distance, 'unit': unit, 'rbf': rbf, 'env': env}
        if name == 'edge_geometry':
            vec, _, _ = ops.edge_geometry(x['positions'], x['edge_index'], x['shifts'], x['cells'], x['edge_frame'], backend=native)
            if 'mask' in x:
                vec = torch.where(x['mask'][:, None], vec, vec.new_tensor([0., 0., 1.]))
            # Native geometry returns sibling custom-autograd outputs. The
            # model's force boundary differentiates w.r.t. vec, not positions:
            # rebuild these two dependents to preserve every radial/angular VJP.
            eps = float(self.config.optional.get('eps', 1e-7))
            if not math.isfinite(eps) or eps <= 0:
                raise ValueError('geometry eps must be positive and finite')
            distance = torch.sqrt(vec.square().sum(-1, keepdim=True) + eps ** 2)
            safe = torch.where(distance > 0, distance, torch.ones_like(distance))
            unit = torch.where(distance > 0, vec / safe, torch.zeros_like(vec))
            return dict(vec=vec, distance=distance, unit=unit)
        if name == 'pair_embedding':
            src, dst = x['edge_index']
            y = x['x'][src] + x['x'][dst]
            return {'y': torch.where(x['mask'][:, None], y, 0.)}
        if name == 'degree_reduce':
            env = torch.where(x['mask'][:, None], x['env'], 0.)
            degree = ops.segment_sum(env.square(), x['edge_index'][1], outputs['degree'].shape[0], backend=native)[:, 0]
            floor = .25 if self.version >= 1.1 else float(self.config.optional.get('eps', 1e-7))
            return {'degree': degree, 'inv_sqrt': torch.rsqrt(degree + floor).reshape(-1, 1, 1)}
        if name == 'radial_fuse':
            return {'y': x['radial'] + x['pair'][:, None, :]}
        if name == 'degree_select':
            return {'y': x['x'][:, :outputs['y'].shape[1]]}
        if name == 'message_reduce':
            message, edge_index, mask = x['message'], x['edge_index'], x['mask']
            n_nodes = outputs['y'].shape[0]
            dst = edge_index[1]
            if 'alpha' in x:
                alpha, gate = x['alpha'], x['gate']
                n_focus, n_heads = alpha.shape[1:]
                if message.shape[-1] % (n_focus * n_heads):
                    raise ValueError('attention heads do not partition the message width')
                head_dim = message.shape[-1] // (n_focus * n_heads)
                values = message.reshape(message.shape[0], message.shape[1], n_focus, n_heads, head_dim)
                weighted = values * alpha[:, None, :, :, None]
                weighted = torch.where(mask[:, None, None, None, None], weighted, 0.)
                reduced = message.new_zeros((n_nodes, message.shape[1], n_focus, n_heads, head_dim))
                reduced = reduced.index_add(0, dst, weighted)
                reduced = reduced * gate[:, None, :, :, None]
                return {'y': reduced.reshape(n_nodes, message.shape[1], message.shape[-1])}
            edge_weight = torch.where(mask[:, None], x['env'], 0.)
            weighted = message * edge_weight[:, None, :]
            reduced = message.new_zeros((n_nodes, message.shape[1], message.shape[2]))
            reduced = reduced.index_add(0, dst, weighted)
            return {'y': reduced * x['inv_sqrt']}
        if name == 'residual':
            return {'y': x['x'] + x['update']}
        if name == 'energy_reduce':
            # Energy accumulation intentionally stays Torch FP64 even on HIP.
            return {'y': ops.segment_sum(x['atom_energy'].to(torch.float64), x['frame_index'], outputs['y'].shape[0])}
        if name == 'edge_gradient':
            if not x['edge_vec'].requires_grad:
                raise RuntimeError('edge vectors require a tracked geometry graph for forces')
            gradient = None
            if x['energy'].requires_grad:
                gradient = torch.autograd.grad(x['energy'].sum(), x['edge_vec'], allow_unused=True, retain_graph=True)[0]
            return {'y': torch.zeros_like(x['edge_vec']) if gradient is None else gradient}
        if name == 'force_virial':
            gradient, vectors = x['gradient'], x['edge_vec']
            if native == 'hip':
                gradient, vectors = gradient.detach(), vectors.detach()
            force, virial = ops.force_virial(gradient, vectors, x['edge_index'], x['edge_frame'], x['mask'],
                                            outputs['force'].shape[0], outputs['virial'].shape[0], backend=native)
            return {'force': force, 'virial': virial.to(torch.float64)}
        raise AssertionError(name)
