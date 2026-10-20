import json
from pathlib import Path
import subprocess
import sys
import unittest


class CLITests(unittest.TestCase):
    def test_infer_has_deepmd_integration_options(self):
        result=subprocess.run([sys.executable,"-m","model","infer","--help"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn("--operator-backend",result.stdout)
        self.assertIn("--neighbor-backend",result.stdout)

    @unittest.skipUnless((Path(__file__).resolve().parents[3] / "deepmd-kit/examples/water/dpa4/lmp/pretrained.pt").is_file(), "requires adjacent water checkpoint")
    def test_pretrained_shape_trace(self):
        checkpoint = Path(__file__).resolve().parents[3] / "deepmd-kit/examples/water/dpa4/lmp/pretrained.pt"
        result = subprocess.run([sys.executable, "-m", "model", "shape", "--checkpoint", str(checkpoint),
            "--atoms", "192", "--edges", "1000", "--frames", "2"], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["outputs"]["force"]["shape"], [192, 3])
        self.assertEqual(report["outputs"]["energy"]["shape"], [2, 1])
        self.assertFalse(report["numerical_prediction"])
        self.assertEqual(report["trace"][-1]["operator"], "force_virial")

    def test_operator_inventory(self):
        result = subprocess.run([sys.executable, "-m", "model", "operators"], capture_output=True, text=True, check=True)
        operators = json.loads(result.stdout)
        self.assertIn("grid_project", [o["name"] for o in operators])
        self.assertIn("edge_gradient", [o["name"] for o in operators])


if __name__ == "__main__":
    unittest.main()
