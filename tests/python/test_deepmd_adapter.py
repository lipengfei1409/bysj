import importlib.util
import inspect
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from torch import nn
from tests.python.upstream_contracts import DEEP, module_with_upstream_forward


class AdapterTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("dcu_dpa4.deepmd"), "missing DeePMD integration package")
        from dcu_dpa4.deepmd import install_operators
        return install_operators

    def test_linear_values_gradients_state_and_instance_isolation(self):
        install = self.api()
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(4, 7), nn.SiLU(), nn.Linear(7, 3)).double()
        other = nn.Linear(4, 7).double()
        original_forward = other.forward
        x = torch.randn(2, 5, 4, dtype=torch.float64).transpose(0, 1).requires_grad_()
        ref = model(x)
        ref_grads = torch.autograd.grad(ref.sum(), (x, *model.parameters()))
        parameters = dict(model.named_parameters())
        keys = tuple(model.state_dict())
        with install(model, operators=("linear",)) as handle:
            out = model(x)
            grads = torch.autograd.grad(out.sum(), (x, *model.parameters()))
            torch.testing.assert_close(out, ref)
            for a,b in zip(grads, ref_grads):
                torch.testing.assert_close(a,b)
            self.assertEqual(tuple(model.state_dict()), keys)
            self.assertTrue(all(dict(model.named_parameters())[k] is v for k,v in parameters.items()))
            self.assertEqual(sum(row["calls"] for row in handle.report()), 2)
            self.assertEqual(other.forward, original_forward)
        self.assertNotIn("forward", model[0].__dict__)
        torch.testing.assert_close(model(x), ref)

    def test_hooks_kwargs_and_exception_restore(self):
        install = self.api()
        model = nn.Linear(4, 3)
        signature=inspect.signature(model.forward)
        calls=[]
        hook=model.register_forward_hook(lambda m,i,o:calls.append(o.shape))
        try:
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with install(model) as handle:
                    self.assertEqual(inspect.signature(model.forward),signature)
                    model(input=torch.ones(2, 4))
                    self.assertEqual(handle.report()[0]["calls"], 1)
                    raise RuntimeError("intentional")
            self.assertEqual(calls,[torch.Size([2,3])])
            self.assertNotIn("forward", model.__dict__)
        finally:
            hook.remove()

    def test_duplicate_install_and_no_matches_fail_without_damage(self):
        install = self.api()
        model=nn.Linear(3,2)
        with install(model):
            with self.assertRaisesRegex(RuntimeError,"already"):
                install(model)
        with self.assertRaisesRegex(ValueError,"matched"):
            install(nn.SiLU())
        self.assertNotIn("forward", model.__dict__)

    def test_unknown_backend_and_operator_fail(self):
        install=self.api()
        with self.assertRaises(ValueError):
            install(nn.Linear(3,2),backend="auto")
        with self.assertRaises(ValueError):
            install(nn.Linear(3,2),operators=("so2",))

    @unittest.skipIf(torch.version.hip, "non-ROCm error path")
    def test_hip_is_not_silently_replaced_by_reference(self):
        with self.assertRaisesRegex(RuntimeError,"HIP|ROCm"):
            self.api()(nn.Linear(3,2),backend="hip")

    def test_empty_batch(self):
        install=self.api()
        model=nn.Linear(4,3)
        with install(model):
            self.assertEqual(model(torch.empty(0,4)).shape,(0,3))

    def test_zero_input_width_linear(self):
        with self.assertWarns(UserWarning):
            model=nn.Linear(0,3)
        x=torch.empty(2,0)
        ref=model(x)
        with self.api()(model):
            torch.testing.assert_close(model(x),ref)

    @unittest.skipUnless(DEEP.is_dir(), "requires upstream source for contract probe")
    def test_upstream_rmsnorm_dtype_and_gradient_contract(self):
        install=self.api()
        model=module_with_upstream_forward("deepmd/pt/model/descriptor/sezm_nn/norm.py","RMSNorm")
        model.dtype=torch.float64;model.eps=1e-7;model.channels=7
        model.register_buffer("eps_tensor",torch.tensor(model.eps,dtype=model.dtype))
        model.adam_scale=nn.Parameter(torch.rand(7,dtype=model.dtype))
        x=torch.randn(2,3,7,dtype=torch.float32,requires_grad=True)
        ref=model(x);gr=torch.autograd.grad(ref.sum(),(x,model.adam_scale))
        with install(model):
            actual=model(x);ga=torch.autograd.grad(actual.sum(),(x,model.adam_scale))
            self.assertEqual(actual.dtype,x.dtype)
            torch.testing.assert_close(actual,ref)
            for a,b in zip(ga,gr):torch.testing.assert_close(a,b)

    @unittest.skipUnless(DEEP.is_dir(), "requires upstream source for contract probe")
    def test_upstream_mlp_residual_timestep_and_dtype_contract(self):
        install=self.api()
        model=module_with_upstream_forward("deepmd/pt/model/network/mlp.py","MLPLayer")
        model.prec=torch.float64;model.resnet=True;model.activate=nn.SiLU()
        model.matrix=nn.Parameter(torch.randn(4,8,dtype=model.prec))
        model.bias=nn.Parameter(torch.randn(8,dtype=model.prec))
        model.idt=nn.Parameter(torch.randn(8,dtype=model.prec))
        x=torch.randn(2,3,4,requires_grad=True)
        ref=model(x);gr=torch.autograd.grad(ref.sum(),(x,*model.parameters()))
        with install(model):
            actual=model(xx=x);ga=torch.autograd.grad(actual.sum(),(x,*model.parameters()))
            torch.testing.assert_close(actual,ref)
            for a,b in zip(ga,gr):torch.testing.assert_close(a,b)

    @unittest.skipUnless(DEEP.is_dir(), "requires upstream source for contract probe")
    def test_upstream_radial_bessel_gaussian_and_gradient_contract(self):
        install=self.api()
        path="deepmd/pt/model/descriptor/sezm_nn/radial.py"
        for family in ("bessel","gaussian"):
            model=module_with_upstream_forward(path,"RadialBasis")
            model.rcut=6.;model.n_radial=6;model.exponent=7;model.dtype=torch.float64;model.basis_family=family
            model.adam_freqs=nn.Parameter(torch.arange(1,7,dtype=model.dtype)[None]*.52)
            model.register_buffer("pi_tensor",torch.tensor(torch.pi,dtype=model.dtype))
            model.register_buffer("gaussian_coeff",torch.tensor(-.5/(6/5)**2,dtype=model.dtype))
            # Non-persistent constants can retain FP32 rounding after .double().
            # Adapters must use the stored values, not reconstruct constants.
            model.pi_tensor=torch.tensor(torch.pi,dtype=torch.float32).double()
            model.gaussian_coeff=torch.tensor(-.5/(6/5)**2,dtype=torch.float32).double()
            env=module_with_upstream_forward(path,"C3CutoffEnvelope")
            import math
            env.p=7;env._series_coefficients=tuple(float(math.comb(k+3,3)) for k in range(7))
            env.register_buffer("rcut_tensor",torch.tensor(6.,dtype=model.dtype))
            model.envelope=env
            x=torch.tensor([[0.],[.001],[2.],[6.]],dtype=model.dtype,requires_grad=True)
            ref=model(x);gr=torch.autograd.grad(ref.sum(),(x,model.adam_freqs))
            with install(model):
                actual=model(r=x);ga=torch.autograd.grad(actual.sum(),(x,model.adam_freqs))
                torch.testing.assert_close(actual,ref,rtol=1e-12,atol=1e-12)
                for a,b in zip(ga,gr):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)

    def test_linear_autocast_matches_torch(self):
        install=self.api()
        model=nn.Linear(4,3)
        x=torch.randn(2,4)
        with torch.autocast("cpu",dtype=torch.bfloat16):
            ref=model(x)
            with install(model):
                actual=model(x)
        self.assertEqual(actual.dtype,ref.dtype)
        torch.testing.assert_close(actual,ref)

    @unittest.skipUnless(DEEP.is_dir(), "requires upstream source for contract probe")
    def test_upstream_channel_linear_noncontiguous_contract(self):
        model=module_with_upstream_forward("deepmd/pt/model/descriptor/sezm_nn/so3.py","ChannelLinear")
        model.weight=nn.Parameter(torch.randn(4,7,dtype=torch.float64))
        model.bias=nn.Parameter(torch.randn(7,dtype=torch.float64));model.use_bias=True
        x=torch.randn(2,3,4,dtype=torch.float64).transpose(0,1).requires_grad_()
        ref=model(x)
        with self.api()(model):
            actual=model(x)
            torch.testing.assert_close(actual,ref)
            ga=torch.autograd.grad(actual.sum(),(x,model.weight),retain_graph=True)
            gr=torch.autograd.grad(ref.sum(),(x,model.weight))
            for a,b in zip(ga,gr):torch.testing.assert_close(a,b)

    @unittest.skipUnless(DEEP.is_dir(), "requires upstream source for contract probe")
    def test_upstream_mlp_strict_promotion_contract(self):
        model=module_with_upstream_forward("deepmd/pt/model/network/mlp.py","MLPLayer",strict=True)
        model.prec=torch.float32;model.resnet=False;model.activate=nn.Identity();model.idt=None
        model.matrix=nn.Parameter(torch.randn(4,7,dtype=torch.float64));model.bias=None
        x=torch.randn(2,4,dtype=torch.float64)
        # Isolate only the global promotion configuration; real forward body
        # and parameter math are not mocked.
        with patch.dict(sys.modules,{"deepmd.pt.utils.env":SimpleNamespace(DP_DTYPE_PROMOTION_STRICT=True)}):
            ref=model(x)
            with self.api()(model):
                torch.testing.assert_close(model(x),ref)


if __name__ == "__main__":
    unittest.main()
