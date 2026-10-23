"""Run with ROCm PyTorch after building dcu_dpa4._C."""
import unittest

import torch

from dcu_dpa4 import ops
from dcu_dpa4.megakernel import (
    DPA4TaskGraph,
    PersistentTaskPlan,
    TaskTemplate,
    build_dpa4_task_graph,
    check_runtime_status,
)
from dcu_dpa4.native import native_available
from model.dpa4.modeling.config import DPA4Config


@unittest.skipUnless(native_available() and torch.cuda.is_available(), "requires the HIP task runtime and a DCU")
class PersistentEdgeFeatureRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(61)

    def _inputs(self, n=23, e=73):
        positions = torch.randn(n, 3, device="cuda", requires_grad=True)
        cells = torch.eye(3, device="cuda").repeat(2, 1, 1) * 6.0
        cells.requires_grad_()
        edge_index = torch.randint(n, (2, e), device="cuda")
        shifts = torch.randint(-1, 2, (e, 3), device="cuda")
        edge_frame = torch.randint(2, (e,), device="cuda")
        mask = torch.rand(e, device="cuda") > 0.15
        frequencies = (torch.arange(1, 7, device="cuda", dtype=torch.float32)[None] * 0.43)
        frequencies.requires_grad_()
        return positions, edge_index, shifts, cells, edge_frame, mask, frequencies

    def _plan(self, edge_count, tile_size, *, chained=False, generation=71):
        if chained:
            graph = DPA4TaskGraph((
                TaskTemplate("edge_geometry", "edge_geometry", "edge", 0,
                             edge_count, tile_size),
                TaskTemplate("radial_basis", "radial_basis", "edge", 0,
                             edge_count, tile_size, ("edge_geometry",)),
            ))
        else:
            graph = build_dpa4_task_graph(
                DPA4Config(channels=4, n_radial=6, lmax=1, n_blocks=0,
                           use_env_seed=False),
                n_atoms=23, edge_capacity=edge_count, n_frames=2,
                edge_tile=tile_size, fuse_edge_features=True,
            )
        target = "radial_basis" if chained else "edge_features"
        return PersistentTaskPlan.from_task_graph(
            graph, target_tasks=target, generation=generation,
        )

    def test_persistent_outputs_and_vjp_match_direct_reference(self):
        args = self._inputs()
        positions, edge_index, shifts, cells, frame, mask, frequencies = args
        plan = self._plan(edge_index.shape[1], tile_size=13, chained=True)
        reference = ops.edge_features(
            positions, edge_index, shifts, cells, frame, mask, frequencies,
            6.0, 7, 5, 1e-7, backend="reference",
        )
        persistent = ops.edge_features_persistent(
            positions, edge_index, shifts, cells, frame, mask, frequencies,
            6.0, 7, 5, 1e-7, runtime_plan=plan,
        )
        check_runtime_status(persistent[-1])
        for expected, actual in zip(reference, persistent[:5]):
            torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)

        upstream = [torch.randn_like(value) for value in reference]
        reference_grads = torch.autograd.grad(reference, (positions, cells, frequencies), upstream)
        persistent_grads = torch.autograd.grad(
            persistent[:5], (positions, cells, frequencies), upstream,
        )
        for expected, actual in zip(reference_grads, persistent_grads):
            torch.testing.assert_close(actual, expected, rtol=7e-4, atol=7e-5)

    def test_empty_edge_plan_returns_empty_outputs_without_worker_launch(self):
        positions, _, _, cells, _, _, frequencies = self._inputs(e=0)
        edge_index = torch.empty(2, 0, dtype=torch.int64, device="cuda")
        shifts = torch.empty(0, 3, dtype=torch.int64, device="cuda")
        frame = torch.empty(0, dtype=torch.int64, device="cuda")
        mask = torch.empty(0, dtype=torch.bool, device="cuda")
        plan = self._plan(0, tile_size=64)
        self.assertEqual(plan.queue_capacity, 0)
        outputs = ops.edge_features_persistent(
            positions, edge_index, shifts, cells, frame, mask, frequencies,
            6.0, runtime_plan=plan,
        )
        check_runtime_status(outputs[-1])
        self.assertEqual(tuple(outputs[0].shape), (0, 3))
        self.assertEqual(tuple(outputs[3].shape), (0, 6))

    def test_generation_reuse_and_current_stream(self):
        args = self._inputs(e=29)
        positions, edge_index, shifts, cells, frame, mask, frequencies = args
        base_stream = torch.cuda.current_stream()
        stream = torch.cuda.Stream()
        stream.wait_stream(base_stream)
        outputs = None
        with torch.cuda.stream(stream):
            plan = self._plan(edge_index.shape[1], tile_size=7, generation=412)
            outputs = ops.edge_features_persistent(
                positions, edge_index, shifts, cells, frame, mask, frequencies,
                6.0, runtime_plan=plan,
            )
        torch.cuda.current_stream().wait_stream(stream)
        check_runtime_status(outputs[-1])
        second = ops.edge_features_persistent(
            positions, edge_index, shifts, cells, frame, mask, frequencies,
            6.0, runtime_plan=self._plan(edge_index.shape[1], tile_size=7, generation=413),
        )
        check_runtime_status(second[-1])
        for left, right in zip(outputs[:5], second[:5]):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
