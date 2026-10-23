"""Run with ROCm PyTorch after building dcu_dpa4._C."""
import unittest

import torch

from dcu_dpa4 import ops
from dcu_dpa4.native import native_available


@unittest.skipUnless(native_available() and torch.cuda.is_available(), "requires built HIP extension and DCU")
class HIPAccuracyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_geometry_forward_and_backward(self):
        n, e = 41, 257
        edge = torch.randint(n, (2, e), device="cuda")
        shift = torch.randint(-1, 2, (e, 3), device="cuda")
        frame = torch.randint(2, (e,), device="cuda")
        x = torch.randn(n, 3, device="cuda")
        cell = torch.eye(3, device="cuda").repeat(2, 1, 1)*8
        inputs = []
        results = []
        for backend in ("reference", "hip"):
            p = x.clone().requires_grad_(); b = cell.clone().requires_grad_()
            out = ops.edge_geometry(p, edge, shift, b, frame, backend=backend)
            upstream = [torch.ones_like(t) for t in out]
            grads = torch.autograd.grad(out, (p, b), upstream)
            results.append(out); inputs.append(grads)
        for ref, actual in zip(results[0]+inputs[0], results[1]+inputs[1]):
            torch.testing.assert_close(actual, ref, rtol=2e-4, atol=2e-4)

    def test_radial_forward_and_backward_near_zero_and_cutoff(self):
        r0 = torch.tensor([0., 1e-6, .3, 2., 5.999, 6., 6.001], device="cuda")[:, None]
        w0 = torch.arange(1, 7, device="cuda", dtype=torch.float32)[None, :]*.52
        results = []
        for backend in ("reference", "hip"):
            r = r0.clone().requires_grad_(); w = w0.clone().requires_grad_()
            rbf, env = ops.radial(r, w, 6., backend=backend)
            grad = torch.autograd.grad((rbf, env), (r, w), (torch.ones_like(rbf), torch.ones_like(env)))
            results.append((rbf, env, *grad))
        for ref, actual in zip(*results):
            torch.testing.assert_close(actual, ref, rtol=2e-4, atol=2e-4)

    def test_radial_exponent_one_at_zero(self):
        results = []
        for backend in ("reference", "hip"):
            r = torch.tensor([[0.], [.1], [6.]], device="cuda", requires_grad=True)
            w = torch.ones(1, 2, device="cuda", requires_grad=True)
            rbf, env = ops.radial(r, w, 6., 1, 1, backend=backend)
            results.append((rbf, env, *torch.autograd.grad((rbf, env), (r, w), (torch.ones_like(rbf), torch.ones_like(env)))))
        for ref, actual in zip(*results):
            torch.testing.assert_close(actual, ref, rtol=2e-4, atol=2e-4)

    def test_fused_edge_features_forward_and_backward(self):
        n, e = 17, 43
        p0 = torch.randn(n, 3, device='cuda')
        cell0 = torch.eye(3, device='cuda').repeat(2, 1, 1) * 5.0
        edge = torch.randint(n, (2, e), device='cuda')
        edge[:, 0] = torch.tensor([0, 0], device='cuda')
        shift = torch.randint(-1, 2, (e, 3), device='cuda')
        frame = torch.randint(2, (e,), device='cuda')
        mask = torch.rand(e, device='cuda') > 0.2
        mask[1] = False
        w0 = torch.arange(1, 7, device='cuda', dtype=torch.float32)[None, :] * 0.52
        reference = native = None
        for backend in ('reference', 'hip'):
            positions = p0.clone().requires_grad_()
            cells = cell0.clone().requires_grad_()
            frequencies = w0.clone().requires_grad_()
            outputs = ops.edge_features(
                positions, edge, shift, cells, frame, mask, frequencies,
                6.0, 7, 5, 1e-7, backend=backend,
            )
            upstream = [torch.randn_like(value) for value in outputs]
            gradients = torch.autograd.grad(outputs, (positions, cells, frequencies), upstream)
            result = (*outputs, *gradients)
            if backend == 'reference':
                reference = result
            else:
                native = result
        for expected, actual in zip(reference, native):
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)

    def test_fused_scalar_features_differentiate_through_edge_vector(self):
        n, e = 13, 37
        p0 = torch.randn(n, 3, device='cuda')
        cell0 = torch.eye(3, device='cuda').repeat(2, 1, 1) * 4.0
        edge = torch.randint(n, (2, e), device='cuda')
        shift = torch.randint(-1, 2, (e, 3), device='cuda')
        frame = torch.randint(2, (e,), device='cuda')
        mask = torch.rand(e, device='cuda') > 0.2
        frequencies = torch.arange(1, 5, device='cuda', dtype=torch.float32)[None, :] * 0.4
        gradients = []
        for backend in ('reference', 'hip'):
            positions = p0.clone().requires_grad_()
            cells = cell0.clone().requires_grad_()
            vec, distance, unit, rbf, env = ops.edge_features(
                positions, edge, shift, cells, frame, mask, frequencies,
                6.0, 5, 4, 1e-7, backend=backend,
            )
            energy = distance.sum() + unit.square().sum() + rbf.sum() + env.sum()
            gradients.append(torch.autograd.grad(energy, vec)[0])
        torch.testing.assert_close(gradients[1], gradients[0], rtol=5e-4, atol=5e-5)

    def test_empty_fused_edge_features(self):
        edge = torch.empty(2, 0, dtype=torch.int64, device='cuda')
        shifts = torch.empty(0, 3, dtype=torch.int64, device='cuda')
        frame = torch.empty(0, dtype=torch.int64, device='cuda')
        mask = torch.empty(0, dtype=torch.bool, device='cuda')
        outputs = ops.edge_features(
            torch.zeros(2, 3, device='cuda'), edge, shifts,
            torch.eye(3, device='cuda')[None], frame, mask,
            torch.ones(1, 6, device='cuda'), 6.0, backend='hip',
        )
        self.assertEqual(tuple(outputs[0].shape), (0, 3))
        self.assertEqual(tuple(outputs[3].shape), (0, 6))

    def test_norm_forward_and_backward(self):
        x0 = torch.randn(129, 17, device="cuda"); w0 = torch.randn(17, device="cuda")
        results = []
        for backend in ("reference", "hip"):
            x = x0.clone().requires_grad_(); w = w0.clone().requires_grad_()
            y = ops.rmsnorm(x, w, backend=backend)
            results.append((y, *torch.autograd.grad(y.sum(), (x, w))))
        for ref, actual in zip(*results):
            torch.testing.assert_close(actual, ref, rtol=2e-4, atol=2e-4)

    def test_linear_and_segment_gradients(self):
        x0 = torch.randn(31, 17, device="cuda"); w0 = torch.randn(17, 11, device="cuda")
        index = torch.randint(7, (31,), device="cuda")
        results = []
        for backend in ("reference", "hip"):
            x=x0.clone().requires_grad_(); w=w0.clone().requires_grad_()
            y=ops.segment_sum(ops.linear(x,w,backend=backend),index,7,backend=backend)
            results.append((y,*torch.autograd.grad(y.sum(),(x,w))))
        for ref, actual in zip(*results):
            torch.testing.assert_close(actual,ref,rtol=2e-4,atol=2e-4)

    def test_linear_tiled_shape_tails_and_bias(self):
        # Non-multiple M/K/N exercises the masked LDS tile path when DUMMA
        # linear is enabled at build time; the default kernel keeps this contract.
        x0 = torch.randn(19, 19, device="cuda")
        w0 = torch.randn(19, 23, device="cuda")
        b0 = torch.randn(23, device="cuda")
        results = []
        for backend in ("reference", "hip"):
            x = x0.clone().requires_grad_()
            w = w0.clone().requires_grad_()
            bias = b0.clone().requires_grad_()
            y = ops.linear(x, w, bias, backend=backend)
            gradients = torch.autograd.grad(y.square().sum(), (x, w, bias))
            results.append((y, *gradients))
        for expected, actual in zip(*results):
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)

        empty_x = torch.empty(2, 0, device="cuda")
        empty_w = torch.empty(0, 23, device="cuda")
        torch.testing.assert_close(
            ops.linear(empty_x, empty_w, b0, backend="hip"), b0.expand(2, -1)
        )

    def test_force_virial_mask_and_multiple_frames(self):
        e,n=257,41
        g=torch.randn(e,3,device="cuda");v=torch.randn_like(g)
        edge=torch.randint(n,(2,e),device="cuda");frame=torch.randint(2,(e,),device="cuda")
        mask=torch.rand(e,device="cuda")>.2
        ref=ops.force_virial(g,v,edge,frame,mask,n,2)
        actual=ops.force_virial(g,v,edge,frame,mask,n,2,backend="hip")
        for a,b in zip(actual,ref):torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-4)

    def test_empty_edges(self):
        x=torch.empty(0,3,device="cuda");edge=torch.empty(2,0,dtype=torch.int64,device="cuda")
        i=torch.empty(0,dtype=torch.int64,device="cuda");s=torch.empty(0,3,dtype=torch.int64,device="cuda")
        for t in ops.edge_geometry(torch.zeros(2,3,device="cuda"),edge,s,torch.eye(3,device="cuda")[None],i,backend="hip"):
            self.assertEqual(t.shape[0],0)
        self.assertEqual(ops.radial(x[:,:1],torch.ones(1,6,device="cuda"),6.,backend="hip")[0].shape,(0,6))

    def test_current_stream(self):
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x=torch.randn(100,16,device="cuda")
            w=torch.ones(16,device="cuda")
            y=ops.rmsnorm(x,w,backend="hip")
            ref=ops.rmsnorm(x,w)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(y,ref,rtol=2e-4,atol=2e-4)


if __name__ == "__main__":
    unittest.main()
