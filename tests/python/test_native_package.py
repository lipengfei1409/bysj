"""Package contracts for the native operator module after the TileLang removal."""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from setuptools import find_packages

ROOT = Path(__file__).resolve().parents[2]


class NativePackageTests(unittest.TestCase):
    def test_kernels_package_is_discovered(self):
        packages = find_packages(str(ROOT / "python"))
        self.assertIn("kernels", packages)
        self.assertTrue((ROOT / "python" / "kernels" / "segment.py").is_file())

    def test_tilelang_ops_package_is_gone(self):
        packages = find_packages(str(ROOT / "python"))
        self.assertNotIn("kernels.tilelang_ops", packages)

    def test_package_imports_without_tilelang_and_exports_the_native_ops(self):
        with patch.dict(sys.modules, {"tilelang": None}):
            kernels = importlib.import_module("kernels")
        self.assertEqual(
            kernels.__all__,
            [
                "blockdiag_available",
        "blockdiag_gemm",
        "blockdiag_gemm_backward",
        "edge_combine",
                "edge_combine_available",
                "force_assembly",
                "gie_available",
                "gie_zonal_scatter",
                "group_by_source",
        "monomial_runs",
        "monomial_runs_available",
        "monomial_runs_backward",
                "rmsnorm",
                "rmsnorm_available",
                "segment_available",
                "segment_offsets",
                "segment_softmax",
                "segment_sum",
                "slot_available",
                "slot_pack_scale",
                "slot_reduce",
        "so2_conv_dmk",
        "so2_conv_dmk_available",
                "zonal_scatter",
            ],
        )
        self.assertFalse(hasattr(kernels, "tilelang_ops"))


if __name__ == "__main__":
    unittest.main()
