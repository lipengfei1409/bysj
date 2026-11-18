"""Opt-in latency comparison: fused segmented HIP kernels vs the ATen chain.

Run on a supported HIP/DTK DCU with::

    DPA4_RUN_SEGMENT_LATENCY=1 PYTHONPATH=python \\
        python -m unittest discover -s tests/csrc/latency -v

The edge/node/channel sizes are the ones the 3000-atom DPA4 trace shows
(181838 edges over 3000 atoms, channel widths 64/256/640), so the printed
speedup is directly comparable with ``docs/task2-hip-fusion-plan.md``.
Each line is a JSON record; set ``DPA4_SEGMENT_LATENCY_JSON`` to also write
them as a JSON array.
"""
from __future__ import annotations

import json
import os
import statistics
import unittest

import torch

from kernels.segment import (
    force_assembly,
    group_by_source,
    segment_available,
    segment_offsets,
    segment_softmax,
    segment_sum,
)
from reference.segment import (
    force_assembly_reference,
    segment_softmax_reference,
    segment_sum_reference,
)

_RUN_LATENCY = os.environ.get("DPA4_RUN_SEGMENT_LATENCY", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_WARMUP = max(0, int(os.environ.get("DPA4_SEGMENT_WARMUP", "10")))
_ITERATIONS = max(1, int(os.environ.get("DPA4_SEGMENT_ITERATIONS", "20")))
_REPEATS = max(3, int(os.environ.get("DPA4_SEGMENT_REPEATS", "5")))

_ATOMS = 3000
_EDGES = 181838
_CHANNELS = (64, 256, 640)


def _median_ms(call, warmup=_WARMUP, iterations=_ITERATIONS, repeats=_REPEATS):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return statistics.median(samples)


@unittest.skipUnless(
    _RUN_LATENCY and segment_available() and torch.cuda.is_available(),
    "set DPA4_RUN_SEGMENT_LATENCY=1 and build the fused-segment HIP extension on a DCU",
)
class SegmentLatencyTests(unittest.TestCase):
    records: list[dict] = []

    @classmethod
    def tearDownClass(cls):
        target = os.environ.get("DPA4_SEGMENT_LATENCY_JSON")
        if target:
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(cls.records, handle, indent=2, sort_keys=True)

    def _record(self, case, fused_ms, eager_ms, extra=None):
        record = {
            "case": case,
            "fused_ms": fused_ms,
            "eager_ms": eager_ms,
            "speedup": eager_ms / fused_ms if fused_ms else float("inf"),
            "atoms": _ATOMS,
            "edges": _EDGES,
        }
        if extra:
            record.update(extra)
        type(self).records.append(record)
        print("segment latency: " + json.dumps(record, sort_keys=True), flush=True)

    @classmethod
    def _grouped_graph(cls, atoms=_ATOMS, edges=_EDGES):
        dst = torch.sort(torch.randint(0, atoms, (edges,), device="cuda")).values
        offsets = segment_offsets(dst, atoms)
        return dst, offsets

    def test_segment_sum_against_eager_chain(self):
        dst, offsets = self._grouped_graph()
        node_scale = torch.rand(_ATOMS, device="cuda") + 0.5
        for channels in _CHANNELS:
            with self.subTest(channels=channels):
                src = torch.randn(_EDGES, channels, device="cuda")

                def eager():
                    # fill_ + index_add_ + mul_: the three traced launches.
                    out = torch.zeros((_ATOMS, channels), device="cuda")
                    out.index_add_(0, dst, src)
                    out.mul_(node_scale.unsqueeze(1))
                    return out

                fused = _median_ms(lambda: segment_sum(src, dst, offsets, None, node_scale))
                reference = _median_ms(eager)
                self._record("segment_sum", fused, reference, {"channels": channels})
                # The fused kernel accumulates in ascending edge order while
                # index_add_ sums with row atomics, so the two disagree by fp32
                # summation order over a ~60-edge segment: ~1e-6 absolute on a
                # sum of magnitude ~10, and an arbitrarily large *relative*
                # error wherever the sum cancels. Check against a float64 ground
                # truth with an fp32 error budget instead of the atomic baseline.
                expected = torch.zeros(
                    (_ATOMS, channels), dtype=torch.float64, device=src.device
                )
                expected.index_add_(0, dst, src.double())
                expected = (expected * node_scale.double().unsqueeze(1)).float()
                budget = 1e-5 * float(expected.abs().max())
                torch.testing.assert_close(
                    segment_sum(src, dst, offsets, None, node_scale),
                    expected,
                    rtol=1e-5,
                    atol=budget,
                )

    def test_segment_softmax_against_eager_chain(self):
        dst, offsets = self._grouped_graph()
        env = torch.rand(_EDGES, device="cuda") + 0.1
        for channels in (64, 256):
            with self.subTest(channels=channels):
                logit = torch.randn(_EDGES, channels, device="cuda")
                null_logit = torch.randn(_ATOMS, channels, device="cuda")
                fused = _median_ms(lambda: segment_softmax(logit, env, offsets, null_logit))
                reference = _median_ms(
                    lambda: segment_softmax_reference(logit, env, dst, offsets, null_logit)
                )
                self._record("segment_softmax", fused, reference, {"channels": channels})

    def test_force_assembly_against_eager_chain(self):
        dst, offsets = self._grouped_graph()
        src = torch.randint(0, _ATOMS, (_EDGES,), device="cuda")
        src_order, src_offsets = group_by_source(src, _ATOMS)
        grad_edge = torch.randn(_EDGES, 3, device="cuda")
        edge_vec = torch.randn(_EDGES, 3, device="cuda")
        fused = _median_ms(
            lambda: force_assembly(grad_edge, edge_vec, src, dst, offsets, src_offsets, src_order)
        )
        reference = _median_ms(
            lambda: force_assembly_reference(grad_edge, edge_vec, src, dst, offsets, src_offsets)
        )
        self._record("force_assembly", fused, reference, {"channels": 3})


if __name__ == "__main__":
    unittest.main()
