import unittest

import torch
import torch.nn.functional as F

from dcu_dpa4.megakernel import DPA4TaskGraph, TaskTemplate, build_dpa4_task_graph
from model.modeling.components import ChargeSpinEmbedding
from model.modeling.config import DPA4Config


class OMol25ConfigTests(unittest.TestCase):
    def test_maps_active_neo_branches(self):
        config = DPA4Config.from_model_params({
            'type_map': ['H', 'O'],
            'descriptor': {
                'rcut': 6.0, 'channels': 32, 'n_radial': 16, 'lmax': 3,
                'mmax': 1, 'n_blocks': 2, 'n_focus': 2, 'n_atten_head': 1,
                'mixing_layers': 3, 'radial_so2_mode': 'degree_channel',
                'radial_so2_rank': 1, 'message_node_so3': True,
                'ffn_so3_grid': True, 'grid_branch': [0, 0, 1],
                'add_chg_spin_ebd': True, 'default_chg_spin': [0.0, 1.0],
                'precision': 'float32',
            },
        })
        self.assertEqual(config.radial_so2_rank, 1)
        self.assertTrue(config.message_node_so3)
        self.assertTrue(config.ffn_so3_grid)
        self.assertEqual(config.grid_branch, (0, 0, 1))
        self.assertTrue(config.charge_spin_enabled)
        self.assertEqual(config.default_charge_spin, (0.0, 1.0))


class ChargeSpinEmbeddingTests(unittest.TestCase):
    def test_matches_frame_condition_formula(self):
        torch.manual_seed(7)
        charge_table = torch.randn(200, 4)
        spin_table = torch.randn(100, 4)
        matrix = torch.randn(8, 4)
        bias = torch.randn(4)
        module = ChargeSpinEmbedding(charge_table, spin_table, matrix, bias)
        conditions = torch.tensor([[0.0, 1.0], [-2.0, 3.0]])
        joined = torch.cat((charge_table[conditions[:, 0].long() + 100],
                            spin_table[conditions[:, 1].long()]), dim=-1)
        torch.testing.assert_close(module(conditions)['y'], F.silu(joined @ matrix + bias))

    def test_rejects_nonintegral_and_out_of_range_conditions(self):
        module = ChargeSpinEmbedding(
            torch.zeros(200, 4), torch.zeros(100, 4), torch.zeros(8, 4), torch.zeros(4)
        )
        for conditions, error in (
            (torch.tensor([[0.5, 1.0]]), 'integers'),
            (torch.tensor([[100.0, 1.0]]), r'\[-100,99\]'),
            (torch.tensor([[0.0, 100.0]]), r'\[0,99\]'),
            (torch.tensor([[float('nan'), 1.0]]), 'finite'),
        ):
            with self.subTest(conditions=conditions), self.assertRaisesRegex(ValueError, error):
                module(conditions)


class DPA4TaskPlannerTests(unittest.TestCase):
    def _config(self):
        return DPA4Config(
            channels=32, n_radial=16, lmax=3, mmax=1, n_blocks=2,
            n_focus=2, n_heads=1, mixing_layers=3,
            radial_so2_mode='degree_channel', radial_so2_rank=1,
            message_node_so3=True, ffn_so3_grid=True, grid_branch=(0, 0, 1),
            charge_spin_enabled=True, default_charge_spin=(0.0, 1.0),
        )

    def test_creates_explicit_reduction_epochs_and_target_branches(self):
        graph = build_dpa4_task_graph(self._config(), n_atoms=7, edge_capacity=769, n_frames=2)
        by_name = {task.name: task for task in graph.tasks}
        self.assertEqual(by_name['edge_geometry'].grid_size, 4)
        self.assertIn('charge_spin_embedding', by_name)
        self.assertIn('block_0_message_grid_project', by_name)
        self.assertIn('block_0_ffn_0_grid_integrate', by_name)
        self.assertGreater(by_name['block_0_message_reduce'].epoch,
                           by_name['block_0_inverse_rotate'].epoch)
        self.assertEqual(by_name['block_0_ffn_0_grid_activation'].variant,
                         'ffn_branches=1;kmax=1')
        self.assertEqual(graph.to_dict()['version'], 1)

    def test_zero_edge_bucket_has_zero_edge_grid(self):
        graph = build_dpa4_task_graph(self._config(), n_atoms=0, edge_capacity=0, n_frames=1)
        edge_task = next(task for task in graph.tasks if task.name == 'edge_geometry')
        self.assertEqual(edge_task.grid_size, 0)

    def test_fused_edge_bucket_replaces_geometry_and_radial_tasks(self):
        graph = build_dpa4_task_graph(
            self._config(), n_atoms=7, edge_capacity=769, n_frames=2,
            fuse_edge_features=True,
        )
        names = {task.name for task in graph.tasks}
        self.assertIn('edge_features', names)
        self.assertNotIn('edge_geometry', names)
        self.assertNotIn('radial_basis', names)
        self.assertIn('radial_mlp', names)

    def test_rejects_cycles(self):
        tasks = (
            TaskTemplate('a', 'op_a', 'node', 0, 1, 1, ('b',)),
            TaskTemplate('b', 'op_b', 'node', 0, 1, 1, ('a',)),
        )
        with self.assertRaisesRegex(ValueError, 'cycle'):
            DPA4TaskGraph(tasks)

class WignerRotationTests(unittest.TestCase):
    def test_packed_blocks_are_orthogonal(self):
        from model.modeling.wigner import WignerRotation

        sign = torch.tensor([-1.0, -1.0, 1.0])
        module = WignerRotation(3, torch.tensor([1, 2, 0]), torch.outer(sign, sign))
        unit = torch.tensor([[0.0, 0.0, 1.0], [0.3, 0.4, 0.8660254]])
        rotation = module(unit)['D']
        eye = torch.eye(16).expand(2, -1, -1)
        torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), eye, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(rotation[0], torch.eye(16), atol=2e-5, rtol=2e-5)

class FittingEnergyTests(unittest.TestCase):
    def test_glu_head_adds_per_type_offsets(self):
        from model.modeling.components import FittingEnergy

        hidden_matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        hidden_bias = torch.tensor([0.1, 0.2, -0.1, 0.3])
        output_matrix = torch.tensor([[0.2], [-0.3]])
        atom_bias = torch.tensor([[1.0], [2.0]])
        output_bias = torch.tensor([[0.5], [0.75]])
        module = FittingEnergy(
            [(hidden_matrix, hidden_bias)], output_matrix, None,
            atom_bias, output_bias, activation_function='silu',
        )
        descriptor = torch.tensor([[0.5, -0.5, 1.0], [1.0, 0.0, -1.0]])
        atype = torch.tensor([0, 1], dtype=torch.int64)
        raw = descriptor @ hidden_matrix + hidden_bias
        value, gate = raw.chunk(2, dim=-1)
        expected = (value * F.silu(gate)) @ output_matrix
        expected = expected + atom_bias[atype] + output_bias[atype]
        torch.testing.assert_close(module(descriptor, atype)['y'], expected.double())

class SO3GridTests(unittest.TestCase):
    def test_neo_so3_projector_shape_and_frame_order(self):
        from model.modeling.so3_grid import SO3GridProjector

        projector = SO3GridProjector(lmax=3, mmax=3, kmax=1, dtype=torch.float32)
        self.assertEqual(projector.frame_set, (0, -1, 1))
        self.assertEqual(projector.to_grid_mat.shape, (152, 48))
        self.assertEqual(projector.from_grid_mat.shape, (48, 152))
        self.assertEqual(projector.frame_zero_index, 0)
