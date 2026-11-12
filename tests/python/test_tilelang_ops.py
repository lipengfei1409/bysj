"""Package-boundary checks for the optional TileLang development area."""

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from setuptools import find_packages

ROOT = Path(__file__).resolve().parents[2]


class TileLangOpsPackageTests(unittest.TestCase):
    def test_package_is_discovered(self):
        packages = find_packages(str(ROOT / "python"))
        self.assertIn("kernels.tilelang_ops", packages)

    def test_package_import_does_not_require_tilelang(self):
        with patch.dict(sys.modules, {"tilelang": None}):
            kernels = importlib.import_module("kernels")
            package = importlib.import_module("kernels.tilelang_ops")
        self.assertEqual(
            kernels.__all__,
            ["gemm", "gie_available", "gie_zonal_scatter", "rmsnorm", "rmsnorm_available", "zonal_scatter"],
        )
        self.assertIn("tilelang_bessel_radial_basis", package.__all__)


if __name__ == "__main__":
    unittest.main()
