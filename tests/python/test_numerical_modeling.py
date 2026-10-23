import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from unittest.mock import patch

from model.dpa4 import modeling
from model.dpa4.modeling.backend import TensorSpec


WATER = Path(__file__).resolve().parents[3] / 'deepmd-kit/examples/water/dpa4/lmp/pretrained.pt'


class ScaleOp(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return {'y': x * self.scale}


class TestOnlyRotation(nn.Module):
    def forward(self, unit):
        d = unit.new_ones(unit.shape[0], 1, 1)
        return {'D': d, 'Dt': d}


class TestOnlySeed(nn.Module):
    def forward(self, scalar, radial, D, env, edge_index, inv_sqrt, mask):
        update = torch.zeros_like(scalar).index_add(0, edge_index[1], radial[:, 0])
        return {'y': (scalar + update)[:, None, None]}


class TestOnlyReadout(nn.Module):
    def forward(self, x):
        return {'y': x[:, 0, 0]}


class TestOnlyEnergy(nn.Module):
    def forward(self, descriptor, atype):
        return {'y': descriptor.square().sum(-1, keepdim=True).double()}


class NumericalModelingTests(unittest.TestCase):
    def backend(self, **kwargs):
        self.assertTrue(hasattr(modeling, 'NumericalBackend'), 'numerical operator registry is missing')
        return modeling.NumericalBackend(modeling.DPA4Config(channels=4, n_radial=2), **kwargs)

    def model(self, dtype=torch.float32):
        self.assertTrue(hasattr(modeling, 'RadialMLP'), 'parameter-owning radial MLP is missing')
        b = self.backend(dtype=dtype)
        b.register('type_embedding', modeling.TypeEmbedding(torch.arange(12, dtype=dtype).reshape(3, 4) / 10))
        b.register('radial_basis', modeling.RadialBasis(b.config, torch.tensor([[1., 2.]], dtype=dtype)))
        b.register('radial_mlp', modeling.RadialMLP([2, 4, 8], lmax=1, channels=4, radial_norm=False).to(dtype))
        return modeling.DPA4Model(b.config, b)

    def batch_graph(self, dtype=torch.float32):
        batch = modeling.AtomicBatch(torch.tensor([[0., 0., 0.], [1., .2, 0.], [0., 0., 0.], [2., 0., 0.]], dtype=dtype),
            torch.tensor([0, 1, 0, 1]), torch.tensor([0, 0, 1, 1]), 2)
        graph = modeling.EdgeGraph(torch.tensor([[1, 0, 3, 0], [0, 1, 2, 1]]),
            torch.zeros(4, 3, dtype=torch.int64), torch.tensor([0, 0, 1, 0]), torch.tensor([True, True, True, False]))
        return batch, graph

    def test_hip_backend_rejects_unsupported_float64_modeling(self):
        with self.assertRaisesRegex(TypeError, 'HIP primitive backend.*float32'):
            self.backend(primitive_backend='hip', dtype=torch.float64)

    def test_registry_stage_override_and_output_contract(self):
        b = self.backend()
        b.register('focus_mix', ScaleOp(2))
        b.register('focus_mix', ScaleOp(3), stage='blocks.1.focus_mix')
        x = torch.ones(2, 4)
        for stage, scale in [('blocks.0.focus_mix', 2), ('blocks.1.focus_mix', 3)]:
            y = b.run('focus_mix', {'x': x}, {'y': TensorSpec((2, 4))}, stage=stage)['y']
            torch.testing.assert_close(y, x * scale)
        with self.assertRaisesRegex(ValueError, 'shape'):
            b.run('focus_mix', {'x': x}, {'y': TensorSpec((2, 5))}, stage='blocks.0.focus_mix')
        with self.assertRaisesRegex(TypeError, 'dtype'):
            b.run('focus_mix', {'x': x}, {'y': TensorSpec((2, 4), torch.float64)}, stage='blocks.0.focus_mix')
        with self.assertRaisesRegex(NotImplementedError, 'wigner_rotation.*geometry.rotation'):
            b.run('wigner_rotation', {'unit': torch.ones(2, 3)}, {}, stage='geometry.rotation')
        with self.assertRaises(KeyError):
            b.register('not_a_dpa4_operator', ScaleOp(1))
        with self.assertRaisesRegex(ValueError, 'registered'):
            b.register('focus_mix', ScaleOp(1))

    def test_radial_mlp_forward_backward_and_no_final_activation(self):
        self.assertTrue(hasattr(modeling, 'RadialMLP'), 'RadialMLP is missing')
        net = modeling.RadialMLP([2, 4, 8], lmax=1, channels=4, radial_norm=False).double()
        x = torch.tensor([[.3, -.2], [.8, .6]], dtype=torch.float64, requires_grad=True)
        actual = net(rbf=x)['y']
        expected = (F.silu(x @ net.net[0].matrix) @ net.net[2].matrix).reshape(2, 2, 4)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.autograd.gradcheck(lambda z: net(rbf=z)['y'], (x,)))
        torch.testing.assert_close(net(rbf=torch.zeros_like(x))['y'], torch.zeros_like(actual))

    def test_real_frontend_mask_dtype_degree_and_registered_parameters(self):
        model = self.model(torch.float64)
        batch, graph = self.batch_graph(torch.float64)
        batch.positions.requires_grad_(True)
        cache = model.prepare_edges(batch, graph)
        self.assertEqual(cache.radial.shape, (4, 2, 4))
        self.assertEqual(cache.radial.dtype, torch.float64)
        torch.testing.assert_close(cache.vec[0], batch.positions[1] - batch.positions[0])
        self.assertEqual(cache.env[-1].item(), 0.)
        self.assertEqual(cache.pair[-1].abs().sum().item(), 0.)
        self.assertEqual(cache.radial[-1].abs().sum().item(), 0.)
        expected_degree = torch.zeros(4, dtype=torch.float64).index_add(0, graph.edge_index[1], cache.env[:, 0].square())
        torch.testing.assert_close(cache.degree, expected_degree)
        torch.testing.assert_close(cache.inv_sqrt, torch.rsqrt(expected_degree + 1e-7).reshape(4, 1, 1))
        torch.testing.assert_close(cache.radial_fused, cache.radial + cache.pair[:, None])
        fused_cache = model.prepare_edges(batch, graph, fuse_edge_features=True)
        for name in ('vec', 'distance', 'unit', 'rbf', 'env', 'radial', 'radial_fused'):
            torch.testing.assert_close(getattr(fused_cache, name), getattr(cache, name))
        self.assertTrue(torch.isfinite(torch.autograd.grad(cache.radial.sum(), batch.positions)[0]).all())
        keys = list(model.state_dict())
        self.assertTrue(any('matrix' in k for k in keys))
        self.assertFalse(any(k.startswith('blocks.') for k in keys), 'backend parameters must have one owner')
        self.assertIn('wigner_rotation', {r['operator'] for r in model.backend.report() if not r['implemented']})
        with self.assertRaisesRegex(NotImplementedError, 'wigner_rotation'):
            model(batch, graph)

    def test_empty_edges_and_invalid_topology(self):
        model = self.model()
        batch, graph = self.batch_graph()
        empty = modeling.EdgeGraph(graph.edge_index[:, :0], graph.shifts[:0], graph.frame_index[:0], graph.mask[:0])
        out = model.prepare_edges(batch, empty)
        self.assertEqual(out.radial.shape, (0, 2, 4))
        self.assertTrue(torch.isfinite(out.inv_sqrt).all())
        for mutate in ('negative', 'cross_frame', 'wrong_frame', 'shift_without_cell', 'bad_type'):
            batch, graph = self.batch_graph()
            if mutate == 'negative':
                graph.edge_index[0, 0] = -1
            elif mutate == 'cross_frame':
                graph.edge_index[0, 0] = 3
            elif mutate == 'wrong_frame':
                graph.frame_index[0] = 1
            elif mutate == 'shift_without_cell':
                graph.shifts[0, 0] = 1
            else:
                batch.atype[0] = 2  # padding row is not an element
            with self.subTest(case=mutate), self.assertRaises(ValueError):
                model.prepare_edges(batch, graph)

    def test_force_frame_reductions_use_existing_operators(self):
        b = self.backend(dtype=torch.float64)
        vectors = torch.tensor([[1., 2., 0.], [0., 1., 0.]], dtype=torch.float64, requires_grad=True)
        energy = vectors.square().sum().reshape(1, 1)
        gradient = b.run('edge_gradient', {'energy': energy, 'edge_vec': vectors}, {'y': TensorSpec((2, 3), torch.float64)}, stage='edge_gradient')['y']
        graph = torch.tensor([[1, 0], [0, 1]])
        out = b.run('force_virial', {'gradient': gradient, 'edge_vec': vectors, 'edge_index': graph,
            'edge_frame': torch.tensor([0, 0]), 'mask': torch.tensor([True, False])},
            {'force': TensorSpec((2, 3), torch.float64), 'virial': TensorSpec((1, 3, 3), torch.float64)}, stage='force_virial')
        torch.testing.assert_close(out['force'], torch.stack([2 * vectors[0], -2 * vectors[0]]))
        torch.testing.assert_close(out['virial'][0], -2 * torch.outer(vectors[0], vectors[0]))
        sums = b.run('energy_reduce', {'atom_energy': torch.tensor([[1.], [2.], [4.]], dtype=torch.float64),
            'frame_index': torch.tensor([0, 1, 1])}, {'y': TensorSpec((2, 1), torch.float64)}, stage='energy_reduce')
        torch.testing.assert_close(sums['y'], torch.tensor([[1.], [6.]], dtype=torch.float64))

    def test_injected_test_graph_energy_force_matches_finite_difference_under_no_grad(self):
        # These test-only modules test wiring, NOT a pretrained DPA4 implementation.
        self.assertTrue(hasattr(modeling, 'NumericalBackend'), 'NumericalBackend is missing')
        c = modeling.DPA4Config(channels=4, n_radial=2, n_blocks=0, lmax=0, use_env_seed=False)
        b = modeling.NumericalBackend(c, dtype=torch.float64)
        b.register('type_embedding', modeling.TypeEmbedding(torch.ones(3, 4, dtype=torch.float64)))
        b.register('radial_basis', modeling.RadialBasis(c, torch.tensor([[1., 2.]], dtype=torch.float64)))
        b.register('radial_mlp', modeling.RadialMLP([2, 4], lmax=0, channels=4).double())
        b.register('wigner_rotation', TestOnlyRotation())
        b.register('geometric_seed', TestOnlySeed())
        b.register('scalar_readout', TestOnlyReadout())
        b.register('fitting_energy', TestOnlyEnergy())
        model = modeling.DPA4Model(c, b).eval()
        batch, graph = self.batch_graph(torch.float64)
        with torch.no_grad():
            out = model(batch, graph)
        self.assertFalse(batch.positions.requires_grad, 'do not mutate caller')
        torch.testing.assert_close(out.force.sum(0), torch.zeros(3, dtype=torch.float64))
        eps = 1e-5
        batch.positions[1, 0] += eps
        plus = model(batch, graph, compute_forces=False).energy.sum().item()
        batch.positions[1, 0] -= 2 * eps
        minus = model(batch, graph, compute_forces=False).energy.sum().item()
        self.assertAlmostEqual(out.force[1, 0].item(), -(plus - minus) / (2 * eps), places=5)

    def test_geometry_derivatives_follow_vector_even_for_sibling_native_outputs(self):
        b = self.backend(dtype=torch.float64)
        batch, graph = self.batch_graph(torch.float64)
        # Native custom-autograd geometry returns sibling outputs: its distance
        # has no autograd edge to its vec. Simulate ONLY that topology locally.
        vec = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        distance = torch.linalg.vector_norm(vec.detach(), dim=-1, keepdim=True).requires_grad_(True)
        unit = (vec.detach() / distance.detach()).requires_grad_(True)
        with patch('model.dpa4.modeling.numerical.ops.edge_geometry', return_value=(vec, distance, unit)):
            result = b.run('edge_geometry', {'positions': batch.positions, 'edge_index': graph.edge_index,
                'shifts': graph.shifts, 'cells': torch.zeros(2, 3, 3, dtype=torch.float64), 'edge_frame': graph.frame_index},
                {'vec': TensorSpec((4, 3), torch.float64), 'distance': TensorSpec((4, 1), torch.float64),
                 'unit': TensorSpec((4, 3), torch.float64)}, stage='edge_geometry')
        gradient = torch.autograd.grad(result['distance'].square().sum(), result['vec'], allow_unused=True)[0]
        self.assertIsNotNone(gradient, 'native distances must not disconnect edge force VJP')
        torch.testing.assert_close(gradient, 2 * vec)

    def test_native_force_assembly_receives_detached_inference_inputs(self):
        b = self.backend(primitive_backend='hip')
        vectors = torch.ones(1, 3, requires_grad=True)
        def assembly(gradient, vectors, *args, **kwargs):
            self.assertFalse(gradient.requires_grad or vectors.requires_grad, 'native force assembly is inference-only')
            return torch.zeros(2, 3), torch.zeros(1, 3, 3)
        with patch('model.dpa4.modeling.numerical.ops.force_virial', side_effect=assembly):
            b.run('force_virial', {'gradient': vectors * 2, 'edge_vec': vectors, 'edge_index': torch.tensor([[1], [0]]),
                'edge_frame': torch.tensor([0]), 'mask': torch.tensor([True])},
                {'force': TensorSpec((2, 3)), 'virial': TensorSpec((1, 3, 3), torch.float64)}, stage='force_virial')

    def test_preserved_composite_interfaces_include_geometry_and_normalization(self):
        from tests.python.test_modeling import meta_batch
        b = modeling.ShapeBackend()
        modeling.DPA4Model(modeling.DPA4Config(), b)(meta_batch(4), edge_capacity=6)
        trace = {t.stage: t for t in b.trace}
        self.assertTrue({'atype', 'edge_vec', 'env', 'mask'}.issubset(trace['environment_seed'].inputs))
        self.assertIn('inv_sqrt', trace['geometric_seed'].inputs)
        self.assertIn('mask', trace['geometric_seed'].inputs)
        self.assertIn('degree', trace['blocks.0.attention_softmax'].inputs)
        self.assertIn('inv_sqrt', trace['blocks.0.message_reduce'].inputs)

    def test_periodic_frontend_and_dtype_migration(self):
        model = self.model().double()
        batch, graph = self.batch_graph(torch.float64)
        batch.cells = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1) * 5
        graph.shifts[0, 0] = -1
        cache = model.prepare_edges(batch, graph)
        torch.testing.assert_close(cache.vec[0], batch.positions[1] - batch.positions[0] + torch.tensor([-5., 0., 0.], dtype=torch.float64))
        self.assertEqual(model.backend.compute_dtype, torch.float64)
        self.assertEqual(cache.radial.dtype, torch.float64)

    def test_radial_norm_contract_matches_upstream_forward(self):
        self.assertTrue(hasattr(modeling, 'RadialMLP'), 'RadialMLP is missing')
        from tests.python.upstream_contracts import module_with_upstream_forward
        actual = modeling.RadialMLP([2, 4, 8], lmax=1, channels=4, radial_norm=True).double()
        x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        oracle_layers = []
        for layer in actual.net:
            if hasattr(layer, 'matrix'):
                oracle = module_with_upstream_forward('deepmd/pt/model/network/mlp.py', 'MLPLayer')
                oracle.matrix, oracle.bias, oracle.idt = layer.matrix, None, None
                oracle.prec, oracle.resnet, oracle.activate = torch.float64, False, nn.Identity()
            elif hasattr(layer, 'adam_scale'):
                oracle = module_with_upstream_forward('deepmd/pt/model/descriptor/sezm_nn/norm.py', 'RMSNorm')
                oracle.adam_scale, oracle.eps, oracle.dtype = layer.adam_scale, layer.eps, torch.float64
                oracle.eps_tensor = torch.tensor(layer.eps, dtype=torch.float64)
                oracle.channels = layer.adam_scale.numel()
            else:
                oracle = layer
            oracle_layers.append(oracle)
        oracle = module_with_upstream_forward('deepmd/pt/model/descriptor/sezm_nn/radial.py', 'RadialMLP')
        oracle.net = nn.Sequential(*oracle_layers)
        expected = oracle(x).reshape(3, 2, 4)
        torch.testing.assert_close(actual(rbf=x)['y'], expected)
        torch.testing.assert_close(torch.autograd.grad(actual(rbf=x)['y'].sum(), x)[0], torch.autograd.grad(expected.sum(), x)[0])

    def test_gelu_matches_deepmd_tanh_approximation(self):
        self.assertTrue(hasattr(modeling, 'RadialMLP'), 'RadialMLP is missing')
        net = modeling.RadialMLP([2, 4, 8], lmax=1, channels=4, radial_norm=False, activation_function='gelu').double()
        with torch.no_grad():
            net.net[0].matrix.fill_(1.)
            net.net[2].matrix.fill_(1.)
        x = torch.tensor([[1., 1.]], dtype=torch.float64)
        expected = (F.gelu(x @ net.net[0].matrix, approximate='tanh') @ net.net[2].matrix).reshape(1, 2, 4)
        torch.testing.assert_close(net(rbf=x)['y'], expected, rtol=1e-12, atol=1e-12)

    def test_state_dict_restores_descriptor_version_semantics(self):
        v11 = self.model(torch.float64)
        v11.backend.version = 1.1
        legacy = self.model(torch.float64)
        legacy.load_state_dict(v11.state_dict(), strict=True)
        self.assertEqual(legacy.backend.version, 1.1)
        batch, graph = self.batch_graph(torch.float64)
        before = v11.prepare_edges(batch, graph)
        after = legacy.prepare_edges(batch, graph)
        torch.testing.assert_close(after.radial, before.radial)
        torch.testing.assert_close(after.inv_sqrt, before.inv_sqrt)

    def test_geometry_uses_checkpoint_eps_and_canonical_padding(self):
        model = self.model(torch.float64)
        model.config.optional['eps'] = .2
        batch, graph = self.batch_graph(torch.float64)
        cache = model.prepare_edges(batch, graph)
        expected = torch.sqrt(cache.vec.square().sum(-1, keepdim=True) + .2 ** 2)
        torch.testing.assert_close(cache.distance, expected)
        torch.testing.assert_close(cache.vec[-1], torch.tensor([0., 0., 1.], dtype=torch.float64))

    @unittest.skipUnless(WATER.is_file(), 'water checkpoint not available')
    def test_water_checkpoint_loads_actual_frontend_weights_only(self):
        self.assertTrue(hasattr(modeling.DPA4Model, 'from_checkpoint'), 'frontend checkpoint factory is missing')
        model = modeling.DPA4Model.from_checkpoint(WATER)
        batch, graph = self.batch_graph()
        cache = model.prepare_edges(batch, graph)
        state = torch.load(WATER, map_location='cpu', weights_only=True)['model']
        prefix = 'model.Default.atomic_model.descriptor.'
        expected_type = state[prefix + 'type_embedding.adam_type_embedding'][batch.atype]
        torch.testing.assert_close(cache.type_feat, expected_type)
        w1 = state[prefix + 'radial_embedding.net.0.matrix']
        w2 = state[prefix + 'radial_embedding.net.2.matrix']
        expected_radial = (F.silu(cache.rbf @ w1) @ w2).reshape(4, 2, 16)
        if float(state.get(prefix + 'version_tensor', torch.tensor(1.)).item()) >= 1.1:
            expected_radial = expected_radial * cache.env[:, :, None]
        torch.testing.assert_close(cache.radial, expected_radial)
        torch.testing.assert_close(cache.inv_sqrt, torch.rsqrt(cache.degree + .25).reshape(4, 1, 1))
        self.assertEqual(len(model.checkpoint_report['loaded_keys']), 4)
        self.assertGreater(len(model.checkpoint_report['deferred_keys']), 0)
        self.assertFalse(model.checkpoint_report['complete_model'])
        with self.assertRaisesRegex(NotImplementedError, 'wigner_rotation'):
            model(batch, graph, compute_forces=False)


if __name__ == '__main__':
    unittest.main()
