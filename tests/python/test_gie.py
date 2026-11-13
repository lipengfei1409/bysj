"""CPU contracts, oracle parity, and build selection for native GIE scatter."""
import importlib
import os
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from kernels.gie import gie_zonal_scatter, zonal_scatter
from reference.gie import zonal_scatter_reference

_gie_module = importlib.import_module("kernels.gie")
_native_module = importlib.import_module("kernels.native")


def _inputs(*, requires_grad=False, degree_shape=(2,)):
    zonal = torch.tensor(
        [[0.2, -0.1, 0.3], [0.4, 0.5, -0.2], [-0.3, 0.6, 0.1], [0.7, -0.2, 0.4]],
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    radial = (torch.arange(24, dtype=torch.float32).reshape(4, 2, 3) / 10).requires_grad_(requires_grad)
    dst = torch.tensor([1, 0, 1, 1], dtype=torch.int64)
    radial_slot = torch.tensor([0, 0, 1], dtype=torch.int64)
    inverse = torch.tensor([0.7, 0.5], dtype=torch.float32).reshape(degree_shape)
    inverse.requires_grad_(requires_grad)
    return zonal, radial, dst, radial_slot, inverse


class GIEBuildConfigurationTests(unittest.TestCase):
    def test_default_build_includes_all_remaining_operators(self):
        root = Path(__file__).resolve().parents[2]
        extension_args = {}

        def fake_cuda_extension(name, **kwargs):
            extension_args["name"] = name
            extension_args.update(kwargs)
            return SimpleNamespace(name=name)

        with patch.dict(os.environ, {}, clear=True), patch.object(
            torch.version, "hip", "6.0.0"
        ), patch(
            "torch.utils.cpp_extension.CUDAExtension",
            side_effect=fake_cuda_extension,
        ), patch("torch.utils.cpp_extension.BuildExtension", return_value="build-ext"), patch(
            "setuptools.setup"
        ):
            runpy.run_path(str(root / "setup.py"), run_name="__main__")

        self.assertEqual(extension_args["name"], "kernels._C")
        self.assertEqual(
            extension_args["sources"],
            [
                "csrc/bindings.cpp",
                "csrc/rmsnorm/rmsnorm.cpp",
                "csrc/gie/gie.cpp",
            ],
        )
        self.assertEqual(extension_args["define_macros"], [])


class GIEReferenceTests(unittest.TestCase):
    def test_reference_matches_segment_sum_and_has_gradients(self):
        inputs = _inputs(requires_grad=True)
        zonal, radial, dst, radial_slot, inverse = inputs
        output = zonal_scatter_reference(*inputs)
        expected = torch.zeros(2, 4, 3, dtype=torch.float32)
        for edge in range(zonal.shape[0]):
            node = int(dst[edge])
            for row, slot in enumerate(radial_slot.tolist()):
                expected[node, row + 1] += (
                    zonal[edge, row]
                    * radial[edge, slot]
                    * inverse.reshape(-1)[node]
                )
        torch.testing.assert_close(output, expected)
        self.assertEqual(output[:, 0, :].count_nonzero().item(), 0)

        upstream = torch.randn_like(output)
        actual_grads = torch.autograd.grad(output, (zonal, radial, inverse), upstream)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in actual_grads))

    def test_reference_accepts_three_dimensional_degree_scale(self):
        inputs = _inputs(degree_shape=(2, 1, 1))
        out_1d = zonal_scatter_reference(*_inputs())
        out_3d = zonal_scatter_reference(*inputs)
        torch.testing.assert_close(out_3d, out_1d)

    def test_reference_empty_edges_returns_zero_output(self):
        zonal = torch.empty((0, 3), dtype=torch.float32)
        radial = torch.empty((0, 2, 4), dtype=torch.float32)
        dst = torch.empty((0,), dtype=torch.int64)
        radial_slot = torch.tensor([0, 0, 1], dtype=torch.int64)
        inverse = torch.ones((2,), dtype=torch.float32)
        output = zonal_scatter_reference(zonal, radial, dst, radial_slot, inverse)
        self.assertEqual(tuple(output.shape), (2, 4, 4))
        self.assertEqual(output.count_nonzero().item(), 0)

    def test_reference_rejects_invalid_indices_and_shapes(self):
        inputs = list(_inputs())
        inputs[2] = torch.tensor([2, 0, 1, 1], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "dst values"):
            zonal_scatter_reference(*inputs)
        inputs = list(_inputs())
        inputs[3] = torch.tensor([0, 2, 1], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "radial_slot values"):
            zonal_scatter_reference(*inputs)
        with self.assertRaisesRegex(ValueError, "zonal must have shape"):
            zonal_scatter_reference(torch.ones(4), *_inputs()[1:])


class GIEWrapperTests(unittest.TestCase):
    def test_invalid_contracts_fail_before_dcu_guard(self):
        valid = _inputs()
        with self.assertRaisesRegex(TypeError, "zonal must be a tensor"):
            zonal_scatter(None, *valid[1:])
        with self.assertRaisesRegex(ValueError, "radial must have shape"):
            zonal_scatter(valid[0], torch.ones(4, 2), *valid[2:])
        with self.assertRaisesRegex(TypeError, "dst and radial_slot must be int64"):
            zonal_scatter(valid[0], valid[1], valid[2].int(), valid[3], valid[4])

    def test_valid_cpu_inputs_reach_dcu_guard(self):
        with patch.object(
            _gie_module, "require_dcu", side_effect=RuntimeError("DCU unavailable")
        ):
            with self.assertRaisesRegex(RuntimeError, "DCU unavailable"):
                zonal_scatter(*_inputs())

    def test_native_dispatch_and_all_value_gradients(self):
        class FakeExtension:
            @staticmethod
            def gie_zonal_scatter_forward(zonal, radial, dst, radial_slot, inverse):
                return zonal_scatter_reference(zonal, radial, dst, radial_slot, inverse)

            @staticmethod
            def gie_zonal_scatter_backward(grad_output, zonal, radial, dst, radial_slot, inverse):
                with torch.enable_grad():
                    z = zonal.detach().requires_grad_(True)
                    r = radial.detach().requires_grad_(True)
                    degree = inverse.detach().requires_grad_(True)
                    output = zonal_scatter_reference(z, r, dst, radial_slot, degree)
                    return torch.autograd.grad(output, (z, r, degree), grad_output)

        inputs = _inputs(requires_grad=True)
        upstream = torch.randn(2, 4, 3)
        expected = zonal_scatter_reference(*inputs)
        expected_grads = torch.autograd.grad(
            expected, (inputs[0], inputs[1], inputs[4]), upstream
        )
        extension = FakeExtension()
        with patch.object(_gie_module, "require_dcu"), patch.object(
            _gie_module, "_extension", return_value=extension
        ):
            actual = gie_zonal_scatter(*inputs)
            actual_grads = torch.autograd.grad(
                actual, (inputs[0], inputs[1], inputs[4]), upstream
            )
        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_loader_requires_gie_entry_point(self):
        extension = SimpleNamespace(
            gie_zonal_scatter_forward=object(), gie_zonal_scatter_backward=object()
        )
        with patch.object(_native_module, "_require_hip_runtime"), patch.object(
            _native_module.importlib, "import_module", return_value=extension
        ):
            self.assertIs(
                _native_module._extension("gie_zonal_scatter_forward"), extension
            )


if __name__ == "__main__":
    unittest.main()
