"""Contracts for the DeepMD-backed DPA4 adapter."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from reference.gie import zonal_scatter_reference
from reference.rmsnorm import rmsnorm as rmsnorm_reference
from reference.so2 import so2_envelope_reduce_reference

from model import dpa4
from model.dpa4 import DPA4Model


class _FakePtExptBackend:
    __module__ = "deepmd.pt_expt.infer.deep_eval"

    def __init__(self, model_tree=None):
        if model_tree is not None:
            self._dpmodel = model_tree


class _FakePtBackend:
    __module__ = "deepmd.pt.infer.deep_eval"

    def __init__(self, model_tree=None):
        if model_tree is not None:
            self._dpmodel = model_tree


class _FakeDeepPot:
    def __init__(self, backend_type=_FakePtExptBackend, model_tree=None):
        self.deep_eval = backend_type(model_tree)
        self.closed = False
        self.result = (np.array([[1.25]]), np.zeros((1, 2, 3)), np.zeros((1, 9)))

    def eval(self, *args, **kwargs):
        self.eval_args = args
        self.eval_kwargs = kwargs
        return self.result

    def get_rcut(self):
        return 6.0

    def get_type_map(self):
        return ["O", "H"]

    def get_ntypes(self):
        return 2

    def get_model_def_script(self):
        return {"model": {"type": "dpa4"}}

    def close(self):
        self.closed = True


class _FakeModelTree:
    def __init__(self, *modules):
        self.child_modules = modules

    def modules(self):
        return iter((self, *self.child_modules))


class _FakeGIE(torch.nn.Module):
    __module__ = "deepmd.dpmodel.descriptor.dpa4_nn.embedding"

    def __init__(self):
        super().__init__()
        self.lmax = 1
        self.channels = 2
        self.radial_slot_index_for_row = np.array([0, 0, 0], dtype=np.int64)
        self.training = False
        self.original_calls = 0

    def call(self, *args, **kwargs):
        self.original_calls += 1
        return "deepmd-gie"

    def forward(self, *args, **kwargs):
        return self.call(*args, **kwargs)


_FakeGIE.__name__ = "GeometricInitialEmbedding"


class _FakeRMSNorm(torch.nn.Module):
    __module__ = "deepmd.dpmodel.descriptor.dpa4_nn.norm"

    def __init__(self):
        super().__init__()
        self.channels = 3
        self.eps = 1.0e-7
        self.precision = "float32"
        self.adam_scale = torch.tensor([0.8, 1.1, 1.3], requires_grad=True)
        self.original_calls = 0

    def call(self, x):
        self.original_calls += 1
        return x + 1.0

    def forward(self, x):
        return self.call(x)


_FakeRMSNorm.__name__ = "RMSNorm"


class _FakeSO2Convolution(torch.nn.Module):
    __module__ = "deepmd.dpmodel.descriptor.dpa4_nn.so2"

    def __init__(self):
        super().__init__()
        self.n_atten_head = 0
        self.precision = "float32"
        self.edge_message_calls = 0
        self.original_forward_calls = 0

    def call(self, x, edge_cache, radial_feat):
        return x

    def edge_message(self, x, edge_cache, radial_feat):
        self.edge_message_calls += 1
        return radial_feat, radial_feat

    def forward_envelope(self, x, edge_cache, radial_feat):
        self.original_forward_calls += 1
        return torch.full_like(x, -1.0)


_FakeSO2Convolution.__name__ = "SO2Convolution"


class DPA4ModelAdapterTests(unittest.TestCase):
    def test_nonempty_tilelang_selector_is_rejected_before_loading(self):
        with mock.patch.object(dpa4, "_create_deep_pot") as create:
            with self.assertRaisesRegex(
                RuntimeError, "TileLang operators are disabled"
            ):
                DPA4Model("water.pt", tilelang_ops=("radial_basis",))
        create.assert_not_called()

    def test_native_default_patches_both_without_mutating_legacy_flags(self):
        gie = _FakeGIE()
        rmsnorm = _FakeRMSNorm()
        fake_model = _FakeDeepPot(model_tree=_FakeModelTree(gie, rmsnorm))
        seen = {}
        legacy_flags = ("DP_TILELANG_DPA4", "DP_TILELANG_RADIAL_BASIS")

        def create_deep_pot(model_file, **kwargs):
            seen["flags"] = {name: os.environ.get(name) for name in legacy_flags}
            seen["model_file"] = model_file
            seen["kwargs"] = kwargs
            return fake_model

        def load_native(name):
            return (lambda: True), mock.Mock(name=f"native_{name}")

        previous = {name: "caller-enabled" for name in legacy_flags}
        with mock.patch.dict(os.environ, previous, clear=False):
            with mock.patch.object(
                dpa4, "_create_deep_pot", side_effect=create_deep_pot
            ), mock.patch.object(
                dpa4, "_load_native_operator", side_effect=load_native
            ) as load_native_mock:
                model = DPA4Model(
                    "water_pt_expt.pt",
                    nlist_backend="native",
                )
            self.assertEqual(
                {name: os.environ.get(name) for name in legacy_flags}, previous
            )

        self.assertEqual(seen["flags"], previous)
        self.assertEqual(seen["model_file"], "water_pt_expt.pt")
        self.assertEqual(seen["kwargs"], {"nlist_backend": "native"})
        self.assertEqual(model.native_ops, ("gie", "rmsnorm"))
        self.assertTrue(model.uses_native_ops)
        self.assertEqual(model.native_ops_patched, {"gie": 1, "rmsnorm": 1})
        self.assertEqual(model.tilelang_ops, ())
        self.assertFalse(model.uses_tilelang)
        self.assertEqual(
            [call.args[0] for call in load_native_mock.call_args_list],
            ["gie", "rmsnorm"],
        )
        self.assertEqual(model.backend, "pt_expt")
        self.assertEqual(model.get_rcut(), 6.0)
        self.assertEqual(model.get_type_map(), ["O", "H"])
        coords = np.zeros((1, 2, 3), dtype=np.float32)
        cells = np.eye(3, dtype=np.float32).reshape(1, 9)
        atom_types = np.array([[0, 1]], dtype=np.int32)
        self.assertIs(model.eval(coords, cells, atom_types), fake_model.result)
        self.assertEqual(fake_model.eval_args, (coords, cells, atom_types))
        model.close()
        self.assertTrue(fake_model.closed)

    def test_so2_envelope_reduction_patches_only_the_supported_reduction_seam(self):
        so2 = _FakeSO2Convolution().eval()
        fake_model = _FakeDeepPot(model_tree=_FakeModelTree(so2))
        with mock.patch.object(
            dpa4, "_create_deep_pot", return_value=fake_model
        ), mock.patch.object(
            dpa4,
            "_load_native_operator",
            return_value=(lambda: True, so2_envelope_reduce_reference),
        ):
            model = DPA4Model("model.pt", native_ops=("so2_envelope_reduce",))

        message = torch.randn(4, 3, 2, requires_grad=True)
        edge_env = torch.tensor([[0.2], [0.5], [0.0], [0.8]], requires_grad=True)
        inverse = torch.tensor([0.5, 0.75, 1.0], requires_grad=True).reshape(3, 1, 1)
        inverse.retain_grad()
        dst = torch.tensor([1, 0, 2, 1], dtype=torch.int64)
        edge_cache = SimpleNamespace(
            edge_env=edge_env,
            edge_src_gate=None,
            inv_sqrt_deg=inverse,
            dst=dst,
        )
        x = torch.randn(3, 3, 2)
        with mock.patch.object(dpa4, "_native_fp32_tensor", return_value=True):
            with mock.patch.object(dpa4, "_native_device_tensor", return_value=True):
                with mock.patch.object(dpa4, "_native_dcu_tensors", return_value=True):
                    with mock.patch.object(
                        dpa4, "_is_fp32_compute_precision", return_value=True
                    ):
                        actual = so2.forward_envelope(x, edge_cache, message)
        expected = so2_envelope_reduce_reference(
            message, edge_env.reshape(-1), dst, inverse
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(so2.edge_message_calls, 1)
        self.assertEqual(so2.original_forward_calls, 0)
        self.assertEqual(model.native_ops_patched, {"so2_envelope_reduce": 1})
        grads = torch.autograd.grad(actual.sum(), (message, edge_env, inverse))
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))

    def test_so2_envelope_reduction_falls_back_for_unsupported_inputs(self):
        so2 = _FakeSO2Convolution().eval()
        fake_model = _FakeDeepPot(model_tree=_FakeModelTree(so2))
        with mock.patch.object(
            dpa4, "_create_deep_pot", return_value=fake_model
        ), mock.patch.object(
            dpa4,
            "_load_native_operator",
            return_value=(lambda: True, so2_envelope_reduce_reference),
        ):
            DPA4Model("model.pt", native_ops=("so2_envelope_reduce",))
        with mock.patch.object(dpa4, "_native_fp32_tensor", return_value=False):
            result = so2.forward_envelope(
                torch.zeros(1, 1, 1), SimpleNamespace(), torch.zeros(0, 1, 1)
            )
        self.assertEqual(result.item(), -1.0)
        self.assertEqual(so2.original_forward_calls, 1)
        self.assertEqual(so2.edge_message_calls, 0)

    def test_empty_native_selection_forces_baseline(self):
        fake_model = _FakeDeepPot(backend_type=_FakePtBackend)
        with mock.patch.object(dpa4, "_create_deep_pot", return_value=fake_model):
            model = DPA4Model("legacy.pt", native_ops=())
        self.assertEqual(model.backend, "pt")
        self.assertEqual(model.native_ops, ())
        self.assertFalse(model.uses_native_ops)
        self.assertFalse(model.uses_tilelang)
        self.assertEqual(model.tilelang_ops, ())

    def test_native_selector_is_validated_before_loading(self):
        with mock.patch.object(dpa4, "_create_deep_pot") as create:
            with self.assertRaisesRegex(ValueError, "unknown native DPA4 operator"):
                DPA4Model("water.pt", native_ops=("not-an-op",))
            with self.assertRaisesRegex(RuntimeError, "eager pt_expt `.pt`"):
                DPA4Model("water.pt2", native_ops=("gie",))
        create.assert_not_called()

    def test_native_selection_rejects_legacy_backend_and_closes_model(self):
        fake_model = _FakeDeepPot(
            backend_type=_FakePtBackend,
            model_tree=_FakeModelTree(_FakeGIE()),
        )
        with mock.patch.object(dpa4, "_create_deep_pot", return_value=fake_model):
            with self.assertRaisesRegex(RuntimeError, "native DPA4 operators require"):
                DPA4Model("legacy.pt", native_ops=("gie",))
        self.assertTrue(fake_model.closed)

    def test_missing_native_target_closes_loaded_model(self):
        fake_model = _FakeDeepPot(model_tree=_FakeModelTree())
        with mock.patch.object(dpa4, "_create_deep_pot", return_value=fake_model):
            with self.assertRaisesRegex(RuntimeError, "no supported module instance"):
                DPA4Model("model.pt", native_ops=("gie",))
        self.assertTrue(fake_model.closed)

    def test_unavailable_native_extension_closes_loaded_model(self):
        fake_model = _FakeDeepPot(model_tree=_FakeModelTree(_FakeGIE(), _FakeRMSNorm()))
        with mock.patch.object(
            dpa4, "_create_deep_pot", return_value=fake_model
        ), mock.patch.object(
            dpa4,
            "_load_native_operator",
            return_value=(lambda: False, mock.Mock()),
        ):
            with self.assertRaisesRegex(RuntimeError, "is unavailable"):
                DPA4Model("model.pt")
        self.assertTrue(fake_model.closed)

    def test_explicit_empty_native_selection_skips_native_loading(self):
        fake_model = _FakeDeepPot()
        with mock.patch.object(
            dpa4, "_create_deep_pot", return_value=fake_model
        ), mock.patch.object(dpa4, "_load_native_operator") as load_native:
            model = DPA4Model("baseline.pt", native_ops=())
        self.assertEqual(model.native_ops, ())
        self.assertFalse(model.uses_native_ops)
        self.assertEqual(model.native_ops_patched, {})
        load_native.assert_not_called()

    def test_native_dispatch_is_instance_local_and_preserves_gradients(self):
        gie = _FakeGIE()
        rmsnorm = _FakeRMSNorm()
        fake_model = _FakeDeepPot(model_tree=_FakeModelTree(gie, rmsnorm))
        other_gie = _FakeGIE()
        other_rmsnorm = _FakeRMSNorm()
        other_model = _FakeDeepPot(
            model_tree=_FakeModelTree(other_gie, other_rmsnorm)
        )
        native_calls = []

        def native_gie(*args):
            native_calls.append("gie")
            return zonal_scatter_reference(*args)

        def native_rmsnorm(x, scale, *, eps):
            native_calls.append("rmsnorm")
            return rmsnorm_reference(x, scale, eps)

        def load_native(name):
            return (lambda: True), {
                "gie": native_gie,
                "rmsnorm": native_rmsnorm,
            }[name]

        with mock.patch.object(
            dpa4, "_create_deep_pot", side_effect=(fake_model, other_model)
        ), mock.patch.object(
            dpa4, "_load_native_operator", side_effect=load_native
        ), mock.patch.object(
            dpa4, "_native_fp32_tensor", return_value=True
        ), mock.patch.object(
            dpa4, "_native_device_tensor", return_value=True
        ), mock.patch.object(
            dpa4, "_native_dcu_tensors", return_value=True
        ), mock.patch.object(
            dpa4, "_is_fp32_compute_precision", return_value=True
        ):
            model = DPA4Model("native.pt", native_ops=("gie", "rmsnorm"))
            baseline_model = DPA4Model("baseline.pt", native_ops=())

            self.assertEqual(model.native_ops_patched, {"gie": 1, "rmsnorm": 1})
            self.assertFalse(baseline_model.uses_native_ops)

            zonal = torch.randn(4, 3, requires_grad=True)
            radial = torch.randn(4, 1, 2, requires_grad=True)
            dst = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
            inverse = torch.tensor(
                [0.7, 0.5, 0.9], dtype=torch.float32
            ).reshape(3, 1, 1).requires_grad_()
            edge_cache = SimpleNamespace(
                dst=dst,
                inv_sqrt_deg=inverse,
                edge_src_gate=None,
                edge_mask=None,
            )
            actual_gie = gie(
                n_nodes=3,
                edge_cache=edge_cache,
                radial_feat=radial,
                zonal_coupling=zonal,
            )
            expected_gie = zonal_scatter_reference(
                zonal,
                radial,
                dst,
                torch.tensor([0, 0, 0], dtype=torch.int64),
                inverse,
            )
            torch.testing.assert_close(actual_gie, expected_gie)
            upstream = torch.randn_like(actual_gie)
            actual_grads = torch.autograd.grad(
                actual_gie, (zonal, radial, inverse), upstream, retain_graph=True
            )
            expected_grads = torch.autograd.grad(
                expected_gie, (zonal, radial, inverse), upstream
            )
            for actual, expected in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual, expected)

            x = torch.randn(5, 3, requires_grad=True)
            actual_norm = rmsnorm(x)
            expected_norm = rmsnorm_reference(
                x, rmsnorm.adam_scale, eps=rmsnorm.eps
            )
            torch.testing.assert_close(actual_norm, expected_norm)
            norm_upstream = torch.randn_like(actual_norm)
            actual_norm_grads = torch.autograd.grad(
                actual_norm,
                (x, rmsnorm.adam_scale),
                norm_upstream,
                retain_graph=True,
            )
            expected_norm_grads = torch.autograd.grad(
                expected_norm, (x, rmsnorm.adam_scale), norm_upstream
            )
            for actual, expected in zip(actual_norm_grads, expected_norm_grads):
                torch.testing.assert_close(actual, expected)

        self.assertEqual(native_calls, ["gie", "rmsnorm"])
        self.assertEqual(gie.original_calls, 0)
        self.assertEqual(rmsnorm.original_calls, 0)
        self.assertEqual(other_gie(), "deepmd-gie")
        self.assertEqual(other_rmsnorm(torch.zeros(1, 3)).shape, (1, 3))

    def test_native_gie_falls_back_for_edge_masks(self):
        gie = _FakeGIE()
        fake_model = _FakeDeepPot(model_tree=_FakeModelTree(gie))
        native_called = []
        with mock.patch.object(
            dpa4, "_create_deep_pot", return_value=fake_model
        ), mock.patch.object(
            dpa4,
            "_load_native_operator",
            return_value=(lambda: True, lambda *args: native_called.append(True)),
        ), mock.patch.object(
            dpa4, "_native_fp32_tensor", return_value=True
        ), mock.patch.object(dpa4, "_native_device_tensor", return_value=True):
            DPA4Model("native.pt", native_ops=("gie",))
            result = gie(
                n_nodes=2,
                edge_cache=SimpleNamespace(
                    dst=torch.tensor([0], dtype=torch.int64),
                    inv_sqrt_deg=torch.ones(2, 1, 1),
                    edge_src_gate=None,
                    edge_mask=torch.ones(1),
                ),
                radial_feat=torch.ones(1, 1, 2),
                zonal_coupling=torch.ones(1, 3),
            )
        self.assertEqual(result, "deepmd-gie")
        self.assertEqual(gie.original_calls, 1)
        self.assertEqual(native_called, [])

    def test_native_rmsnorm_falls_back_on_unsupported_cpu_input(self):
        rmsnorm = _FakeRMSNorm()
        fake_model = _FakeDeepPot(model_tree=_FakeModelTree(rmsnorm))
        with mock.patch.object(
            dpa4, "_create_deep_pot", return_value=fake_model
        ), mock.patch.object(
            dpa4, "_load_native_operator", return_value=(lambda: True, mock.Mock())
        ):
            DPA4Model("native.pt", native_ops=("rmsnorm",))
        result = rmsnorm(torch.zeros(2, 3))
        self.assertTrue(torch.equal(result, torch.ones(2, 3)))
        self.assertEqual(rmsnorm.original_calls, 1)


_RUN_INTEGRATION = os.environ.get("DP_RUN_DPA4_MODEL_INTEGRATION", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@unittest.skipUnless(
    _RUN_INTEGRATION,
    "set DP_RUN_DPA4_MODEL_INTEGRATION=1 and model/input paths "
    "to run DeepMD integration",
)
class DPA4ModelIntegrationTests(unittest.TestCase):
    def test_deepmd_dpa4_energy_and_force(self):
        model_path = os.environ.get("DP_DPA4_MODEL_FILE")
        input_path = os.environ.get("DP_DPA4_INPUT_NPZ")
        if not model_path or not input_path:
            self.fail("set DP_DPA4_MODEL_FILE and DP_DPA4_INPUT_NPZ")
        if not os.path.isfile(model_path) or not os.path.isfile(input_path):
            self.fail("DP_DPA4_MODEL_FILE and DP_DPA4_INPUT_NPZ must exist")

        with np.load(input_path) as sample:
            coords = sample["coords"]
            atom_types = sample["atom_types"]
            cells = sample["cells"] if "cells" in sample else None
        raw_ops = os.environ.get("DP_DPA4_MODEL_NATIVE_OPS")
        ops = None if raw_ops is None else tuple(
            item.strip() for item in raw_ops.split(",") if item.strip()
        )
        with DPA4Model(model_path, native_ops=ops) as model:
            outputs = model.eval(coords, cells, atom_types)

        if isinstance(outputs, dict):
            energy = outputs.get("energy", outputs.get("energy_redu"))
            force = outputs.get("force", outputs.get("energy_derv_r"))
        else:
            energy, force = outputs[:2]
        self.assertIsNotNone(energy, "DeepMD evaluation did not return energy")
        self.assertIsNotNone(force, "DeepMD evaluation did not return force")
        self.assertTrue(np.isfinite(np.asarray(energy)).all())
        self.assertTrue(np.isfinite(np.asarray(force)).all())

    @staticmethod
    def _energy_force_virial(outputs):
        if isinstance(outputs, dict):
            return (
                outputs.get("energy", outputs.get("energy_redu")),
                outputs.get("force", outputs.get("energy_derv_r")),
                outputs.get("virial", outputs.get("energy_derv_c")),
            )
        energy, force = outputs[:2]
        virial = outputs[2] if len(outputs) > 2 else None
        return energy, force, virial

    def test_native_ops_match_baseline_energy_force_and_virial(self):
        model_path = os.environ.get("DP_DPA4_MODEL_FILE")
        input_path = os.environ.get("DP_DPA4_INPUT_NPZ")
        if not model_path or not input_path:
            self.fail("set DP_DPA4_MODEL_FILE and DP_DPA4_INPUT_NPZ")
        if not os.path.isfile(model_path) or not os.path.isfile(input_path):
            self.fail("DP_DPA4_MODEL_FILE and DP_DPA4_INPUT_NPZ must exist")

        with np.load(input_path) as sample:
            coords = sample["coords"]
            atom_types = sample["atom_types"]
            cells = sample["cells"] if "cells" in sample else None

        variants = ((), ("gie",), ("rmsnorm",), ("gie", "rmsnorm"))
        extra_ops = tuple(
            item.strip()
            for item in os.environ.get("DP_DPA4_MODEL_EXTRA_NATIVE_OPS", "").split(",")
            if item.strip()
        )
        if extra_ops and extra_ops not in variants:
            variants = (*variants, extra_ops)
        results = {}
        native_loader = dpa4._load_native_operator
        for native_ops in variants:
            native_calls = {name: 0 for name in native_ops}

            def load_native(name):
                available, implementation = native_loader(name)

                def tracked_implementation(*args, **kwargs):
                    native_calls[name] += 1
                    return implementation(*args, **kwargs)

                return available, tracked_implementation

            with mock.patch.object(
                dpa4, "_load_native_operator", side_effect=load_native
            ):
                with DPA4Model(
                    model_path,
                    native_ops=native_ops,
                ) as model:
                    self.assertEqual(set(model.native_ops_patched), set(native_ops))
                    outputs = model.eval(coords, cells, atom_types)
            for name, count in native_calls.items():
                self.assertGreater(
                    count,
                    0,
                    f"native operator {name!r} was patched but never dispatched",
                )
            results[native_ops] = self._energy_force_virial(outputs)

        baseline = results[()]
        for native_ops in variants[1:]:
            actual = results[native_ops]
            for label, expected_value, actual_value in zip(
                ("energy", "force", "virial"), baseline, actual
            ):
                with self.subTest(native_ops=native_ops, output=label):
                    if expected_value is None or actual_value is None:
                        self.assertIs(expected_value, actual_value)
                    else:
                        np.testing.assert_allclose(
                            np.asarray(actual_value),
                            np.asarray(expected_value),
                            rtol=2.0e-4,
                            atol=2.0e-5,
                            err_msg=f"native_ops={native_ops} changed {label}",
                        )
