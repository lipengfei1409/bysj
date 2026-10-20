import unittest

import torch

from model.modeling import AtomicBatch, DPA4Config, DPA4Model, ShapeBackend
from dcu_dpa4.catalog import OPERATORS


def meta_batch(n=192, f=1):
    return AtomicBatch(torch.empty(n, 3, device="meta"),
        torch.empty(n, dtype=torch.int64, device="meta"),
        torch.empty(n, dtype=torch.int64, device="meta"), f,
        torch.empty(f, 3, 3, device="meta"))


class ModelingTests(unittest.TestCase):
    def test_radial_modulation_precedes_so2_mixing(self):
        backend = ShapeBackend()
        DPA4Model(DPA4Config(), backend)(meta_batch(10), edge_capacity=20)
        names = [t.operator for t in backend.trace]
        self.assertLess(names.index("radial_modulation"), names.index("so2_linear"))

    def test_checkpoint_norm_flags_and_zero_mixing(self):
        backend = ShapeBackend()
        DPA4Model(DPA4Config(mixing_layers=0, sandwich_norm=(False, False, False, False)), backend)(meta_batch(10), edge_capacity=20)
        names = [t.operator for t in backend.trace]
        self.assertNotIn("equivariant_norm", names)
        self.assertNotIn("so2_linear", names)
        self.assertIn("radial_modulation", names)

    def test_full_energy_force_shape_path(self):
        backend = ShapeBackend()
        model = DPA4Model(DPA4Config(), backend)
        output = model(meta_batch(), edge_capacity=1000)
        self.assertEqual(output.energy.shape, (1, 1))
        self.assertEqual(output.atom_energy.shape, (192, 1))
        self.assertEqual(output.force.shape, (192, 3))
        self.assertEqual(output.virial.shape, (1, 3, 3))
        self.assertEqual(output.energy.device.type, "meta")
        self.assertEqual(backend.trace[0].operator, "neighbor_graph")
        self.assertEqual(backend.trace[-1].operator, "force_virial")
        self.assertTrue(all(t.operator in OPERATORS for t in backend.trace))

    def test_empty_edges_and_multiple_frames(self):
        backend = ShapeBackend()
        output = DPA4Model(DPA4Config(n_heads=0, use_env_seed=False), backend)(meta_batch(12, 2), edge_capacity=0)
        self.assertEqual(output.virial.shape, (2, 3, 3))
        self.assertNotIn("attention_logits", [t.operator for t in backend.trace])
        self.assertNotIn("environment_seed", [t.operator for t in backend.trace])

    def test_placeholder_fails_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "neighbor_graph"):
            DPA4Model(DPA4Config())(meta_batch(), edge_capacity=5)

    def test_scheduled_blocks_and_focus_shapes(self):
        config = DPA4Config(n_blocks=2, l_schedule=(2, 1), m_schedule=(1, 1), n_focus=2, focus_dim=8, n_heads=2)
        backend = ShapeBackend()
        DPA4Model(config, backend)(meta_batch(10), edge_capacity=30)
        rotated = [t.outputs["y"].shape for t in backend.trace if t.operator == "gather_rotate"]
        self.assertEqual(rotated, [(30, 2, 7, 8), (30, 2, 4, 8)])
        self.assertIn("degree_select", [t.operator for t in backend.trace])

    def test_invalid_head_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            DPA4Config(focus_dim=7, n_heads=2)

    def test_schedule_defaults_match_deepmd(self):
        params = {"type_map": ["H"], "descriptor": {"rcut": 6., "channels": 16, "n_radial": 6,
            "l_schedule": [2, 1], "mmax": None}}
        config = DPA4Config.from_model_params(params)
        self.assertEqual(config.orders, (2, 1))
        params["descriptor"]["mmax"] = 2
        self.assertEqual(DPA4Config.from_model_params(params).orders, (2, 1))
        params["descriptor"]["l_schedule"] = []
        self.assertEqual(DPA4Config.from_model_params(params).n_blocks, 0)

    def test_increasing_degree_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-increasing"):
            DPA4Config(n_blocks=2, l_schedule=(1, 2), m_schedule=(1, 1))


if __name__ == "__main__":
    unittest.main()
