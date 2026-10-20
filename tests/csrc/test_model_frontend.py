"""Opt-in real DCU frontend tests, not complete pretrained E/F/V validation."""
import os
from pathlib import Path
import unittest

import torch

from model.modeling import AtomicBatch, EdgeGraph, DPA4Model


@unittest.skipUnless(os.environ.get('DPA4_RUN_MODEL_NATIVE_TESTS') == '1',
                     'set DPA4_RUN_MODEL_NATIVE_TESTS=1 with water checkpoint on DCU')
class NativeFrontendTests(unittest.TestCase):
    def setUp(self):
        from dcu_dpa4.native import native_available
        if not torch.cuda.is_available() or not native_available():
            raise RuntimeError('explicit native frontend test requires DCU and built HIP extension')
        self.path = Path(os.environ.get('DPA4_WATER_CHECKPOINT',
            str(Path(__file__).resolve().parents[3] / 'deepmd-kit/examples/water/dpa4/lmp/pretrained.pt')))
        if not self.path.is_file():
            raise RuntimeError('set DPA4_WATER_CHECKPOINT to the trusted water pretrained.pt')

    def run_frontend(self, primitive, linear):
        model = DPA4Model.from_checkpoint(self.path, primitive_backend=primitive, linear_backend=linear).to('cuda')
        positions = torch.tensor([[0., 0., 0.], [.96, .1, 0.], [-.24, .93, .1]], device='cuda', requires_grad=True)
        batch = AtomicBatch(positions, torch.tensor([0, 1, 1], device='cuda'),
            torch.zeros(3, dtype=torch.int64, device='cuda'), 1)
        edge = torch.tensor([[1, 2, 0, 2, 0, 1, 0], [0, 0, 1, 1, 2, 2, 0]], device='cuda')
        graph = EdgeGraph(edge, torch.zeros(7, 3, dtype=torch.int64, device='cuda'),
            torch.zeros(7, dtype=torch.int64, device='cuda'),
            torch.tensor([True] * 6 + [False], device='cuda'))
        features = model.prepare_edges(batch, graph)
        loss = features.radial.square().sum() + features.unit[:, 0].sum()
        gradient = torch.autograd.grad(loss, positions)[0]
        return features, gradient

    def compare(self, linear):
        expected, grad_expected = self.run_frontend('reference', 'reference')
        actual, grad_actual = self.run_frontend('hip', linear)
        for name in vars(expected):
            with self.subTest(tensor=name):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(grad_actual, grad_expected, rtol=5e-4, atol=5e-5)

    def test_hip_frontend_values_and_coordinate_gradient(self):
        self.compare('hip')

    @unittest.skipUnless(os.environ.get('DPA4_RUN_ROCBLAS_TESTS') == '1',
                         'also set DPA4_RUN_ROCBLAS_TESTS=1 with rocBLAS extension')
    def test_rocblas_frontend_values_and_coordinate_gradient(self):
        self.compare('rocblas')
