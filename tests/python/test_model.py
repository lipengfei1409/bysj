import os
import unittest
from pathlib import Path

from model.dpa4.checkpoint import load_checkpoint_spec
from model.dpa4.stages import build_stage_plan


EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "deepmd-kit/examples/water/dpa4/lmp/pretrained.pt"
)


@unittest.skipUnless(EXAMPLE.is_file(), "DeePMD water checkpoint is unavailable")
class CheckpointTests(unittest.TestCase):
    def test_water_checkpoint_metadata(self):
        spec = load_checkpoint_spec(EXAMPLE)
        self.assertEqual(spec.model_type, "dpa4")
        self.assertEqual(spec.type_map, ("O", "H"))
        self.assertEqual(spec.rcut, 6.0)
        self.assertEqual(spec.n_radial, 6)
        self.assertEqual(spec.n_blocks, 1)
        self.assertEqual(spec.radial_frequencies_shape, (1, 6))

    def test_stage_plan_covers_model_and_force(self):
        spec = load_checkpoint_spec(EXAMPLE)
        stages = build_stage_plan(spec)
        names = [stage.name for stage in stages]
        self.assertEqual(names[0], "neighbor_graph")
        self.assertIn("interaction_block_0", names)
        self.assertEqual(names[-1], "force_virial")
        self.assertLess(names.index("fitting_energy"), names.index("force_virial"))


if __name__ == "__main__":
    unittest.main()
