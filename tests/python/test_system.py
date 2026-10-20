import unittest
from pathlib import Path

import numpy as np

from model.system import load_deepmd_system


EXAMPLE = Path(__file__).resolve().parents[3] / "deepmd-kit/examples/water/data/data_0"


@unittest.skipUnless(EXAMPLE.is_dir(), "DeePMD water data is unavailable")
class SystemTests(unittest.TestCase):
    def test_water_data_has_consistent_shapes(self):
        system = load_deepmd_system(EXAMPLE, max_frames=2)
        self.assertEqual(system.coords.shape[0], 2)
        self.assertEqual(system.coords.shape[2], 3)
        self.assertEqual(system.cells.shape, (2, 3, 3))
        self.assertEqual(system.energy.shape, (2, 1))
        self.assertEqual(system.force.shape, system.coords.shape)
        self.assertEqual(system.atype.shape, (system.coords.shape[1],))
        self.assertEqual(system.type_map, ("O", "H"))

    def test_coordinates_and_labels_are_finite(self):
        system = load_deepmd_system(EXAMPLE, max_frames=1)
        self.assertTrue(np.isfinite(system.coords).all())
        self.assertTrue(np.isfinite(system.energy).all())
        self.assertTrue(np.isfinite(system.force).all())


if __name__ == "__main__":
    unittest.main()
