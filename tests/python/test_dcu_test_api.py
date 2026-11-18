"""Host guard against stale API references inside the DCU-only tests.

``tests/csrc`` and ``tests/dcu`` only execute on a DCU, so a call to a name this
project has removed stays invisible to the CPU suite until someone runs it on the
cluster. That is exactly how ``fused_deepmd.install()`` survived its rename to
``deepmd_patch()``: the host tests were updated, the DCU parity test was not, and
it failed only on the GPU host.

This module scans those trees for the names removed in this cleanup and imports
every DCU test module, so both a stale name and an import-time break surface in
the CPU run.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DCU_TREES = (ROOT / "tests" / "csrc", ROOT / "tests" / "dcu")

# Names removed from the project; a hit in a DCU test means the test would fail
# on the cluster and could not be caught locally.
REMOVED_NAMES = {
    "fused_deepmd.install": "renamed to fused_deepmd.deepmd_patch",
    "_install_native_ops": "removed with the pt_expt-only native patcher",
    "_load_native_operator": "removed with the pt_expt-only native patcher",
    "_make_gie_call": "removed with the pt_expt-only native patcher",
    "_make_rmsnorm_call": "removed with the pt_expt-only native patcher",
    "_is_dpa4_gie": "removed with the pt_expt-only native patcher",
    "_is_plain_rmsnorm": "removed with the pt_expt-only native patcher",
    "_make_so2_envelope_reduce": "replaced by the fused envelope seam",
    "_normalize_native_ops": "removed with the native_ops selector",
    "_normalize_tilelang_ops": "removed with the TileLang package",
    "native_available": "removed with the GEMM wrapper",
    "so2_envelope_reduce": "folded into segment_sum / fused=(\"envelope\",)",
    "kernels.gemm": "removed; no build provides it",
    "native_ops=": "removed; use fused= instead",
    "tilelang_ops=": "removed; TileLang dispatch is gone",
}


def _python_files():
    for tree in DCU_TREES:
        if tree.is_dir():
            yield from sorted(tree.rglob("*.py"))


class RemovedNameTests(unittest.TestCase):
    def test_dcu_trees_do_not_reference_removed_names(self):
        offences = []
        for path in _python_files():
            text = path.read_text(encoding="utf-8")
            for name, hint in REMOVED_NAMES.items():
                for number, line in enumerate(text.splitlines(), start=1):
                    if name in line:
                        offences.append(
                            f"{path.relative_to(ROOT)}:{number} uses {name!r} ({hint})"
                        )
        self.assertEqual(offences, [], "\n".join(offences))


class DcuTestImportTests(unittest.TestCase):
    def test_every_dcu_test_module_imports_on_the_host(self):
        loaded = []
        for path in _python_files():
            if not path.name.startswith("test_"):
                continue
            module_name = "dcu_import_probe_" + path.stem
            spec = importlib.util.spec_from_file_location(module_name, path)
            self.assertIsNotNone(spec, f"cannot build a spec for {path}")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as error:
                # An optional third-party dependency (DeepMD, triton, rocblas)
                # is absent on this host; the import path itself is fine.
                self.assertIn(
                    error.name,
                    {"deepmd", "triton", "rocblas", "tilelang"},
                    f"{path.relative_to(ROOT)} failed to import: {error}",
                )
            else:
                loaded.append(path.name)
            finally:
                import sys

                sys.modules.pop(module_name, None)
        self.assertTrue(loaded, "no DCU test module imported successfully")


if __name__ == "__main__":
    unittest.main()
