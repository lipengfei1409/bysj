"""Strict, partial checkpoint loading; never claims a complete DPA4 model."""
from __future__ import annotations

import math
import torch

from ..checkpoint import load_checkpoint_spec
from .config import DPA4Config
from .components import AttentionLogits, AttentionSoftmax, ChargeSpinEmbedding, DegreeLinear, EnvironmentSeed, EquivariantNorm, Film, FittingEnergy, FocusCompetition, GatedActivation, GatherRotate, GeometricSeed, InverseRotate, RadialSO2, SO2Linear, ScalarReadout, TensorResult, TypeEmbedding, RadialBasis, RadialMLP
from .so3_grid import EquivariantFFN, SO3GridNet
from .wigner import WignerRotation
from .numerical import NumericalBackend


def load_frontend_checkpoint(path, *, head=None, primitive_backend='reference', linear_backend='reference'):
    """Read trusted .pt weights. Deferred keys are reported, not discarded silently."""
    spec = load_checkpoint_spec(path, head)
    config = DPA4Config.from_model_params(spec.model_params)
    d = config.optional
    if d.get('basis_type', 'bessel').removesuffix('/fix') != 'bessel':
        raise NotImplementedError('frontend checkpoint loader currently supports Bessel only')
    unsupported = [key for key in ('use_spin', 'exclude_types') if d.get(key)]
    unsupported += [key for key in ('inner_clamp_r_inner', 'inner_clamp_r_outer') if d.get(key) is not None]
    if unsupported:
        raise NotImplementedError(f'frontend conditioning not implemented: {unsupported}')
    raw = torch.load(spec.path, map_location='cpu', weights_only=True)
    state = raw.get('model', raw)
    keys = [k for k in state if k.endswith('.radial_basis.adam_freqs') and (head is None or f'.{head}.' in k)]
    if len(keys) != 1:
        raise ValueError('checkpoint descriptor prefix is ambiguous')
    prefix = keys[0].removesuffix('radial_basis.adam_freqs')
    # Folding LoRA requires upstream-specific transforms; never silently omit it.
    if any('lora' in k.lower() for k in state if k.startswith(prefix)):
        raise NotImplementedError('fold LoRA with DeePMD before loading frontend weights')
    frequencies = state[keys[0]]
    version_value = state.get(prefix + 'version_tensor', torch.tensor(1.0))
    version = float(version_value.item())
    if version > 1.2:
        raise NotImplementedError(f'unknown descriptor version {version}')
    backend = NumericalBackend(config, primitive_backend=primitive_backend, dtype=frequencies.dtype, version=version)
    type_key = prefix + 'type_embedding.adam_type_embedding'
    table = state[type_key]
    if table.shape != (len(config.type_map) + 1, config.channels) or table.dtype != frequencies.dtype:
        raise ValueError('checkpoint type table shape/dtype does not match descriptor')
    radial_norm = d.get('edge_norm', True)
    if isinstance(radial_norm, (list, tuple)):
        if len(radial_norm) != 3:
            raise ValueError('edge_norm requires three flags')
        radial_norm = radial_norm[0]
    hidden = [config.channels if int(v) == 0 else int(v) for v in d.get('radial_mlp', [0])]
    radial_mlp = RadialMLP([config.n_radial, *hidden, (config.initial_l + 1) * config.channels],
        lmax=config.initial_l, channels=config.channels, radial_norm=bool(radial_norm),
        activation_function=d.get('activation_function', 'silu'),
        linear_backend=linear_backend, norm_backend=primitive_backend).to(frequencies.dtype)
    radial_prefix = prefix + 'radial_embedding.'
    radial_state = {k[len(radial_prefix):]: v for k, v in state.items() if k.startswith(radial_prefix)}
    if any(v.dtype != frequencies.dtype for v in radial_state.values() if isinstance(v, torch.Tensor)):
        raise ValueError('radial MLP weights must share frequency dtype')
    radial_mlp.load_state_dict(radial_state, strict=True)
    backend.register('type_embedding', TypeEmbedding(table))
    backend.register('radial_basis', RadialBasis(config, frequencies, backend=primitive_backend))
    backend.register('radial_mlp', radial_mlp)
    loaded = {type_key, keys[0], *(radial_prefix + k for k in radial_state)}
    if config.charge_spin_enabled:
        charge_prefix = prefix + 'charge_spin_embedding.'
        charge_keys = {
            'charge_embedding.adam_type_embedding': charge_prefix + 'charge_embedding.adam_type_embedding',
            'spin_embedding.adam_type_embedding': charge_prefix + 'spin_embedding.adam_type_embedding',
            'mix_layer.matrix': charge_prefix + 'mix_layer.matrix',
            'mix_layer.bias': charge_prefix + 'mix_layer.bias',
        }
        missing = [key for key in charge_keys.values() if key not in state]
        if missing:
            raise ValueError(f'checkpoint is missing charge/spin weights: {missing}')
        charge_module = ChargeSpinEmbedding(
            state[charge_keys['charge_embedding.adam_type_embedding']],
            state[charge_keys['spin_embedding.adam_type_embedding']],
            state[charge_keys['mix_layer.matrix']],
            state[charge_keys['mix_layer.bias']],
            activation_function=d.get('activation_function', 'silu'),
            backend=linear_backend,
        ).to(frequencies.dtype)
        backend.register('charge_spin_embedding', charge_module)
        loaded.update(charge_keys.values())
    if config.use_env_seed:
        env_prefix = prefix + 'env_seed_embedding.'
        env_state = {key[len(env_prefix):]: value for key, value in state.items() if key.startswith(env_prefix)}
        env_type_key = 'env_type_embed.adam_type_embedding'
        if env_type_key not in env_state:
            raise ValueError('checkpoint is missing the environment type embedding')
        env_module = EnvironmentSeed(
            len(config.type_map),
            config.n_radial,
            config.channels,
            env_type_table=env_state['env_type_embed.adam_type_embedding'],
            activation_function=d.get('activation_function', 'silu'),
            backend=linear_backend,
            bias=bool(d.get('mlp_bias', False)),
            eps=float(d.get('eps', 1e-7)),
        ).to(frequencies.dtype)
        env_module.load_state_dict(env_state, strict=True)
        backend.register('environment_seed', env_module)
        loaded.update(env_prefix + key for key in env_state)

        edge_norm = d.get('edge_norm', True)
        film_norm = bool(edge_norm[1]) if isinstance(edge_norm, (list, tuple)) else bool(edge_norm)
        film_prefix = prefix
        film_suffixes = {
            'film_scale_norm.adam_scale': 'film_scale_norm.adam_scale',
            'film_shift_norm.adam_scale': 'film_shift_norm.adam_scale',
            'film_scale_strength_log': 'film_scale_strength_log',
            'film_shift_strength_log': 'film_shift_strength_log',
        }
        film_keys = {name: film_prefix + suffix for name, suffix in film_suffixes.items()}
        required_film_keys = [film_keys['film_scale_strength_log'], film_keys['film_shift_strength_log']]
        if film_norm:
            required_film_keys.extend((
                film_keys['film_scale_norm.adam_scale'],
                film_keys['film_shift_norm.adam_scale'],
            ))
        missing = [key for key in required_film_keys if key not in state]
        if missing:
            raise ValueError(f'checkpoint is missing FiLM weights: {missing}')
        if film_norm:
            scale_norm = state[film_keys['film_scale_norm.adam_scale']]
            shift_norm = state[film_keys['film_shift_norm.adam_scale']]
        else:
            scale_norm = shift_norm = None
        film_module = Film(
            scale_norm,
            shift_norm,
            state[film_keys['film_scale_strength_log']],
            state[film_keys['film_shift_strength_log']],
            eps=float(d.get('eps', 1e-7)),
            backend=primitive_backend,
        ).to(frequencies.dtype)
        backend.register('film', film_module)
        loaded.update(required_film_keys)
    wigner_prefix = prefix + 'wigner_calc.'
    wigner_state = {key[len(wigner_prefix):]: value for key, value in state.items() if key.startswith(wigner_prefix)}
    required_wigner_keys = {'l1_perm', 'l1_sign_outer'}
    if not required_wigner_keys.issubset(wigner_state):
        missing = sorted(required_wigner_keys - set(wigner_state))
        raise ValueError(f'checkpoint is missing Wigner basis buffers: {missing}')
    wigner_module = WignerRotation(
        config.initial_l,
        wigner_state['l1_perm'],
        wigner_state['l1_sign_outer'],
        eps=float(d.get('eps', 1e-7)),
    )
    wigner_module.load_state_dict(wigner_state, strict=True)
    backend.register('wigner_rotation', wigner_module)
    loaded.update(wigner_prefix + key for key in wigner_state)
    gie_prefix = prefix + 'gie.'
    gie_state = {key[len(gie_prefix):]: value for key, value in state.items() if key.startswith(gie_prefix)}
    required_gie_keys = {
        'non_scalar_row_index',
        'zonal_m0_col_index_for_row',
        'radial_slot_index_for_row',
    }
    if not required_gie_keys.issubset(gie_state):
        missing = sorted(required_gie_keys - set(gie_state))
        raise ValueError(f'checkpoint is missing GIE index buffers: {missing}')
    gie_module = GeometricSeed(
        config.initial_l,
        config.channels,
        gie_state['non_scalar_row_index'],
        gie_state['zonal_m0_col_index_for_row'],
        gie_state['radial_slot_index_for_row'],
    )
    gie_module.load_state_dict(gie_state, strict=True)
    backend.register('geometric_seed', gie_module)
    loaded.update(gie_prefix + key for key in gie_state)
    grid_mlp_config = d.get('grid_mlp', False)
    if isinstance(grid_mlp_config, (list, tuple)):
        if len(grid_mlp_config) != 3:
            raise ValueError('grid_mlp must be a bool or three branch flags')
        message_grid_mlp, ffn_grid_mlp = bool(grid_mlp_config[1]), bool(grid_mlp_config[2])
    else:
        message_grid_mlp = ffn_grid_mlp = bool(grid_mlp_config)
    for block in range(config.n_blocks):
        block_model_prefix = prefix + f'blocks.{block}.'
        block_prefix = block_model_prefix + 'so2_conv.'
        norm_sites = []
        if config.sandwich_norm[0]:
            norm_sites.append(('pre_so2_norm', 'pre_so2_norm'))
        if config.sandwich_norm[1]:
            norm_sites.append(('post_so2_norm', 'post_so2_norm'))
        if config.sandwich_norm[2]:
            norm_sites.extend((f'pre_ffn_norm.{index}', f'pre_ffn_norms.{index}')
                              for index in range(config.ffn_blocks))
        if config.sandwich_norm[3]:
            norm_sites.extend((f'post_ffn_norm.{index}', f'post_ffn_norms.{index}')
                              for index in range(config.ffn_blocks))
        for stage_name, checkpoint_name in norm_sites:
            norm_prefix = block_model_prefix + checkpoint_name + '.'
            norm_state = {key[len(norm_prefix):]: value for key, value in state.items()
                          if key.startswith(norm_prefix)}
            required = {'adam_scale', 'bias', 'expand_index', 'balance_weight'}
            missing = sorted(required - set(norm_state))
            if missing:
                raise ValueError(f'checkpoint is missing {checkpoint_name} tensors in block {block}: {missing}')
            norm_module = EquivariantNorm(
                norm_state['adam_scale'], norm_state['bias'],
                norm_state['expand_index'], norm_state['balance_weight'],
                eps=float(d.get('eps', 1e-7)),
            ).to(frequencies.dtype)
            norm_module.load_state_dict(norm_state, strict=True)
            backend.register('equivariant_norm', norm_module,
                             stage=f'blocks.{block}.{stage_name}')
            loaded.update(norm_prefix + key for key in norm_state)
        for layer_name, operator in (('pre_focus_mix', 'focus_mix'), ('post_focus_mix', 'focus_unmix')):
            weight_key = block_prefix + layer_name + '.weight'
            expand_key = block_prefix + layer_name + '.expand_index'
            required = (weight_key, expand_key)
            missing = [key for key in required if key not in state]
            if missing:
                raise ValueError(f'checkpoint is missing {layer_name} weights for block {block}: {missing}')
            bias_key = block_prefix + layer_name + '.bias'
            bias = state.get(bias_key)
            if bool(d.get('mlp_bias', False)) and bias is None:
                raise ValueError(f'checkpoint is missing {layer_name} bias for block {block}')
            module = DegreeLinear(
                state[weight_key], state[expand_key], bias,
                output_singleton_focus=layer_name == 'post_focus_mix',
            ).to(frequencies.dtype)
            backend.register(operator, module, stage=f'blocks.{block}.{operator}')
            loaded.update(required)
            if bias is not None:
                loaded.add(bias_key)
        backend.register(
            'gather_rotate',
            GatherRotate(config.degrees[block], config.orders[block], config.n_focus, config.focus_channels),
            stage=f'blocks.{block}.gather_rotate',
        )
        backend.register(
            'inverse_rotate',
            InverseRotate(config.degrees[block], config.orders[block], config.n_focus, config.focus_channels),
            stage=f'blocks.{block}.inverse_rotate',
        )
        if config.radial_so2_mode != 'none':
            radial_block_prefix = prefix + f'blocks.{block}.so2_conv.'
            radial_state = {}
            for component in ('radial_hidden_proj', 'radial_degree_mixer'):
                component_prefix = radial_block_prefix + component + '.'
                for key, value in state.items():
                    if key.startswith(component_prefix):
                        radial_state[key[len(radial_block_prefix):]] = value
            mixer_weight_key = 'radial_degree_mixer.weight'
            compact_key = 'radial_degree_mixer.kernel_compact_index'
            dense_key = 'radial_degree_mixer.kernel_dense_index'
            missing = [key for key in (mixer_weight_key, compact_key, dense_key) if key not in radial_state]
            if missing:
                raise ValueError(f'checkpoint is missing radial SO(2) weights for block {block}: {missing}')
            basis = radial_state.get('radial_degree_mixer.channel_basis')
            if config.radial_so2_mode == 'degree_channel' and config.radial_so2_rank > 0 and basis is None:
                raise ValueError(f'checkpoint is missing radial SO(2) channel basis for block {block}')
            radial_module = RadialSO2(
                config.degrees[block],
                config.orders[block],
                config.channels,
                config.n_focus,
                config.focus_channels,
                config.radial_so2_mode,
                config.radial_so2_rank,
                radial_state[mixer_weight_key],
                radial_state[compact_key],
                radial_state[dense_key],
                channel_basis=basis,
                hidden_projection_weight=radial_state.get('radial_hidden_proj.weight'),
            ).to(frequencies.dtype)
            radial_module.load_state_dict(radial_state, strict=True)
            backend.register('radial_so2', radial_module, stage=f'blocks.{block}.radial_so2')
            loaded.update(radial_block_prefix + key for key in radial_state)
        for layer in range(config.mixing_layers):
            line_prefix = block_prefix + f'so2_linears.{layer}.'
            line_state = {key[len(line_prefix):]: value for key, value in state.items() if key.startswith(line_prefix)}
            needed = {'weight_m0', 'm0_idx', 'pos_indices', 'neg_indices'}
            needed.update(f'weight_m.{order - 1}' for order in range(1, config.orders[block] + 1))
            missing = sorted(needed - set(line_state))
            if missing:
                raise ValueError(f'checkpoint is missing SO(2) layer {layer} weights for block {block}: {missing}')
            bias0 = line_state.get('bias0')
            if bool(d.get('mlp_bias', False)) and bias0 is None:
                raise ValueError(f'checkpoint is missing SO(2) layer {layer} scalar bias for block {block}')
            line_module = SO2Linear(
                config.degrees[block],
                config.orders[block],
                config.n_focus,
                config.focus_channels,
                line_state['weight_m0'],
                [line_state[f'weight_m.{order}'] for order in range(config.orders[block])],
                line_state['m0_idx'],
                line_state['pos_indices'],
                line_state['neg_indices'],
                bias0=bias0,
            ).to(frequencies.dtype)
            line_module.load_state_dict(line_state, strict=True)
            backend.register('so2_linear', line_module, stage=f'blocks.{block}.so2_linear.{layer}')
            loaded.update(line_prefix + key for key in line_state)
        s2_activation = d.get('s2_activation')
        so2_grid_activation = bool(s2_activation[0]) if isinstance(s2_activation, (list, tuple)) and len(s2_activation) == 2 else bool(s2_activation)
        if so2_grid_activation:
            raise NotImplementedError('SO(2) grid activation is not implemented in the independent model path')
        for layer in range(max(0, config.mixing_layers - 1)):
            nonlinearity_prefix = block_prefix + f'non_linearities.{layer}.'
            nonlinearity_state = {
                key[len(nonlinearity_prefix):]: value for key, value in state.items()
                if key.startswith(nonlinearity_prefix)
            }
            required = {'expand_index', 'gate_linear.weight'}
            missing = sorted(required - set(nonlinearity_state))
            if missing:
                raise ValueError(f'checkpoint is missing SO(2) activation weights for block {block}, layer {layer}: {missing}')
            gate_bias = nonlinearity_state.get('gate_linear.bias')
            if bool(d.get('mlp_bias', False)) and gate_bias is None:
                raise ValueError(f'checkpoint is missing SO(2) gate bias for block {block}, layer {layer}')
            activation_module = GatedActivation(
                config.degrees[block],
                config.orders[block],
                config.focus_channels,
                config.n_focus,
                nonlinearity_state['expand_index'],
                nonlinearity_state['gate_linear.weight'],
                activation_function=d.get('activation_function', 'silu'),
                gate_bias=gate_bias,
            ).to(frequencies.dtype)
            activation_module.load_state_dict(nonlinearity_state, strict=True)
            backend.register('gated_activation', activation_module,
                             stage=f'blocks.{block}.so2_activation.{layer}')
            loaded.update(nonlinearity_prefix + key for key in nonlinearity_state)
        if config.focus_compete and config.n_focus > 1:
            focus_prefix = block_prefix
            focus_weight_key = focus_prefix + 'adamw_focus_compete_w'
            focus_norm_key = focus_prefix + 'focus_compete_norm.adam_scale'
            focus_bias_key = focus_prefix + 'focus_compete_bias'
            if focus_weight_key not in state:
                raise ValueError(f'checkpoint is missing focus competition weights for block {block}')
            focus_norm = state.get(focus_norm_key) if config.focus_norm else None
            if config.focus_norm and focus_norm is None:
                raise ValueError(f'checkpoint is missing focus competition norm for block {block}')
            focus_bias = state.get(focus_bias_key)
            if bool(d.get('mlp_bias', False)) and focus_bias is None:
                raise ValueError(f'checkpoint is missing focus competition bias for block {block}')
            focus_module = FocusCompetition(
                state[focus_weight_key], config.n_focus, config.focus_channels,
                norm_scale=focus_norm, bias=focus_bias,
                eps=float(d.get('eps', 1e-7)),
            ).to(frequencies.dtype)
            backend.register('focus_compete', focus_module, stage=f'blocks.{block}.focus_compete')
            loaded.add(focus_weight_key)
            if focus_norm is not None:
                loaded.add(focus_norm_key)
            if focus_bias is not None:
                loaded.add(focus_bias_key)
        if config.n_heads:
            if d.get('atten_v_proj', False) or d.get('atten_o_proj', False):
                raise NotImplementedError('attention value/output projections are not implemented')
            attention_names = (
                'attn_qk_norm.adam_scale',
                'attn_q_proj.weight',
                'attn_k_proj.weight',
                'attn_output_gate_norm.adam_scale',
                'adamw_attn_logit_w',
                'adamw_attn_gate_w',
            )
            attention_state = {name: state[block_prefix + name] for name in attention_names
                               if block_prefix + name in state}
            missing = [name for name in attention_names if name not in attention_state]
            z_bias_key = block_prefix + 'adamw_attn_z_bias_raw'
            if z_bias_key not in state:
                missing.append('adamw_attn_z_bias_raw')
            if missing:
                raise ValueError(f'checkpoint is missing attention weights for block {block}: {missing}')
            attention_focuses = 1 if config.atten_f_mix else config.n_focus
            attention_channels = config.n_focus * config.focus_channels if config.atten_f_mix else config.focus_channels
            attention_module = AttentionLogits(
                attention_focuses, attention_channels, config.n_heads,
                attention_state, eps=float(d.get('eps', 1e-7)),
            ).to(frequencies.dtype)
            attention_module.load_state_dict(attention_state, strict=True)
            softmax_module = AttentionSoftmax(
                state[z_bias_key], eps=float(d.get('eps', 1e-7))
            ).to(frequencies.dtype)
            backend.register('attention_logits', attention_module,
                             stage=f'blocks.{block}.attention_logits')
            backend.register('attention_softmax', softmax_module,
                             stage=f'blocks.{block}.attention_softmax')
            loaded.update(block_prefix + name for name in attention_state)
            loaded.add(z_bias_key)
        if config.message_node_so3:
            if config.grid_branch[1] > 0:
                message_op = 'branch'
            else:
                message_op = 'mlp' if message_grid_mlp else 'glu'
            message_grid_prefix = block_prefix + 'message_node_grid_product.'
            message_grid_state = {
                key[len(message_grid_prefix):]: value for key, value in state.items()
                if key.startswith(message_grid_prefix)
            }
            message_grid = SO3GridNet(
                lmax=config.degrees[block], channels=config.focus_channels,
                n_focus=config.n_focus, mode='cross', op_type=message_op,
                dtype=frequencies.dtype, layout='flat', kmax=config.kmax,
                grid_branches=max(1, config.grid_branch[1]), mlp_bias=config.mlp_bias,
                residual_scale_init=1e-3, trainable=bool(d.get('trainable', True)),
            )
            message_grid.load_state_dict(message_grid_state, strict=True)
            backend.register('message_node_so3', TensorResult(message_grid, ('query', 'context')),
                             stage=f'blocks.{block}.message_node_so3')
            loaded.update(message_grid_prefix + key for key in message_grid_state)
        if config.ffn_so3_grid:
            hidden_channels = config.ffn_hidden or 32 * ((8 * config.channels + 95) // 96)
            ffn_prefix = block_model_prefix + 'ffns.'
            for layer in range(config.ffn_blocks):
                layer_prefix = ffn_prefix + f'{layer}.'
                ffn_state = {
                    key[len(layer_prefix):]: value for key, value in state.items()
                    if key.startswith(layer_prefix)
                }
                lmax = config.degrees[block] + config.extra_node_l
                ffn = EquivariantFFN(
                    lmax=lmax, channels=config.channels, hidden_channels=hidden_channels,
                    dtype=frequencies.dtype, kmax=config.kmax,
                    grid_mlp=ffn_grid_mlp, grid_branch=config.grid_branch[2],
                    ffn_so3_grid=True, mlp_bias=config.mlp_bias,
                    trainable=bool(d.get('trainable', True)),
                )
                ffn.load_state_dict(ffn_state, strict=True)
                backend.register('so3_grid_ffn', TensorResult(ffn), stage=f'blocks.{block}.ffn.{layer}')
                loaded.update(layer_prefix + key for key in ffn_state)
    if config.so3_readout != 'mlp':
        raise NotImplementedError('the independent OMol readout currently requires so3_readout="mlp"')
    if config.readout_layers != 1:
        raise NotImplementedError('multiple SO(3) readout layers are not yet wired')
    readout_prefix = prefix + 'output_ffn.'
    readout_state = {
        key[len(readout_prefix):]: value for key, value in state.items()
        if key.startswith(readout_prefix)
    }
    readout_lmax = (config.degrees[-1] if config.degrees else config.lmax) + config.extra_node_l
    readout_hidden = config.ffn_hidden or 32 * ((8 * config.channels + 95) // 96)
    output_ffn = EquivariantFFN(
        lmax=readout_lmax, channels=config.channels, hidden_channels=readout_hidden,
        dtype=frequencies.dtype, kmax=config.kmax,
        grid_mlp=True, grid_branch=0, ffn_so3_grid=True,
        mlp_bias=config.mlp_bias, trainable=bool(d.get('trainable', True)),
    )
    output_ffn.load_state_dict(readout_state, strict=True)
    backend.register('scalar_readout', ScalarReadout(output_ffn))
    loaded.update(readout_prefix + key for key in readout_state)

    fit_config = spec.model_params.get('fitting_net', {})
    unsupported_fit = [name for name in ('numb_fparam', 'numb_aparam', 'dim_case_embd', 'case_film_embd', 'use_aparam_as_mask')
                       if fit_config.get(name)]
    if unsupported_fit:
        raise NotImplementedError(f'fitting conditioning is not implemented: {unsupported_fit}')
    if not bool(fit_config.get('mixed_types', True)):
        raise NotImplementedError('per-type fitting networks are not implemented')
    atomic_prefix = prefix.removesuffix('descriptor.')
    fitting_prefix = atomic_prefix + 'fitting_net.'
    hidden_prefix = fitting_prefix + 'filter_layers.networks.0.hidden_layers.'
    hidden_indices = sorted({
        int(key[len(hidden_prefix):].split('.', 1)[0])
        for key in state
        if key.startswith(hidden_prefix) and key.endswith('.linear.matrix')
    })
    hidden_layers = []
    for index in hidden_indices:
        layer_prefix = hidden_prefix + f'{index}.linear.'
        matrix_key, bias_key = layer_prefix + 'matrix', layer_prefix + 'bias'
        if matrix_key not in state or bias_key not in state:
            raise ValueError(f'fitting GLU layer {index} is missing its matrix/bias')
        hidden_layers.append((state[matrix_key], state[bias_key]))
        loaded.update((matrix_key, bias_key))
    if hidden_indices != list(range(len(hidden_indices))):
        raise ValueError('fitting hidden layer indices must be contiguous from zero')
    output_prefix = fitting_prefix + 'filter_layers.networks.0.output_layer.'
    output_matrix_key = output_prefix + 'matrix'
    if output_matrix_key not in state:
        raise ValueError('checkpoint is missing the fitting output matrix')
    output_bias_key = output_prefix + 'bias'
    output_bias = state.get(output_bias_key)
    bias_atom_key = fitting_prefix + 'bias_atom_e'
    out_bias_key = atomic_prefix + 'out_bias'
    if bias_atom_key not in state or out_bias_key not in state:
        raise ValueError('checkpoint is missing atom-energy bias/output statistics')
    fitting_module = FittingEnergy(
        hidden_layers,
        state[output_matrix_key],
        output_bias,
        state[bias_atom_key],
        state[out_bias_key],
        activation_function=fit_config.get('activation_function', d.get('activation_function', 'silu')),
        backend=linear_backend,
    ).to(frequencies.dtype)
    backend.register('fitting_energy', fitting_module)
    loaded.update((output_matrix_key, bias_atom_key, out_bias_key))
    if output_bias is not None:
        loaded.add(output_bias_key)
    backend.requires_grad_(bool(d.get('trainable', True)))
    if d.get('basis_type', '').endswith('/fix'):
        backend.operator_modules['radial_basis'].adam_freqs.requires_grad_(False)
    from .dpa4 import DPA4Model
    model = DPA4Model(config, backend).eval()
    # version is read as control metadata, not a trained tensor.
    model.checkpoint_report = {
        'path': str(spec.path), 'descriptor_prefix': prefix, 'version': version,
        'loaded_keys': sorted(loaded),
        'deferred_keys': sorted(k for k in state if isinstance(state[k], torch.Tensor) and k not in loaded and k != prefix + 'version_tensor'),
        'complete_model': False,
        'scope': 'OMol25 active reference composition is wired in code; checkpoint/runtime validation is pending and HIP persistent scheduling is not implemented',
    }
    return model
