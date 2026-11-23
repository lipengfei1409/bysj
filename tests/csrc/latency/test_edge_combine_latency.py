"""Opt-in latency comparison: C6 edge_combine vs the eager broadcast chain.

Run on a supported HIP/DTK DCU with::

    DPA4_RUN_EDGE_COMBINE_LATENCY=1 PYTHONPATH=python \\
        python -m unittest discover -s tests/csrc/latency -p 'test_edge_combine*' -v

The shape is the profiled edge stream (181838 edges, 9 l-rows, C = 64). The
eager chain is ``(full + residual) * scale + gate`` -- three kernels, three
passes; ``edge_combine`` folds the scale into the coefficients and does one.
"""
from __future__ import annotations

import json
import os
import statistics
import unittest

import torch

from kernels.combine import edge_combine, edge_combine_available

_RUN = os.environ.get("DPA4_RUN_EDGE_COMBINE_LATENCY", "0").lower() in {
    "1", "true", "yes", "on"
}
_WARMUP = max(0, int(os.environ.get("DPA4_EDGE_COMBINE_WARMUP", "10")))
_ITERATIONS = max(1, int(os.environ.get("DPA4_EDGE_COMBINE_ITERATIONS", "20")))
_REPEATS = max(3, int(os.environ.get("DPA4_EDGE_COMBINE_REPEATS", "5")))

_EDGES = 181838
_ROWS = 9
_CHANNELS = 64


def _median_ms(call):
    for _ in range(_WARMUP):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_ITERATIONS):
            call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / _ITERATIONS)
    return statistics.median(samples)


@unittest.skipUnless(
    _RUN and edge_combine_available() and torch.cuda.is_available(),
    "set DPA4_RUN_EDGE_COMBINE_LATENCY=1 on a built DCU host",
)
class EdgeCombineLatencyTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cuda").manual_seed(4)
        self.full = torch.randn(_EDGES, _ROWS, _CHANNELS, device="cuda", generator=generator)
        self.residual = torch.randn(_EDGES, _ROWS, _CHANNELS, device="cuda", generator=generator)
        self.gate = torch.randn(_EDGES, _ROWS, 1, device="cuda", generator=generator)

    def _record(self, case, eager_ms, fused_ms):
        record = {
            "case": case,
            "edges": _EDGES,
            "rows": _ROWS,
            "channels": _CHANNELS,
            "eager_ms": eager_ms,
            "fused_ms": fused_ms,
            "speedup": eager_ms / fused_ms,
        }
        print(f"edge combine latency: {json.dumps(record, sort_keys=True)}", flush=True)
        path = os.environ.get("DPA4_EDGE_COMBINE_LATENCY_JSON")
        if path:
            records = []
            if os.path.exists(path):
                with open(path) as handle:
                    try:
                        records = json.load(handle)
                    except json.JSONDecodeError:
                        records = []
            records.append(record)
            with open(path, "w") as handle:
                json.dump(records, handle, indent=2, sort_keys=True)

    def test_three_op_chain_against_one_fused_pass(self):
        scale = 0.5

        def chain():
            return (self.full + self.residual) * scale + self.gate

        def fused():
            return edge_combine(
                self.full, self.residual, self.gate,
                coefficients=(scale, scale, 1.0),
            )

        expected = chain()
        actual = fused()
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-6, atol=2e-6))
        eager_ms = _median_ms(chain)
        fused_ms = _median_ms(fused)
        self._record("add+mul+broadcast_add", eager_ms, fused_ms)
        self.assertGreater(eager_ms / fused_ms, 1.0)

    def test_two_term_accumulate(self):
        def chain():
            return self.full * 0.25 + self.residual * 0.75

        def fused():
            return edge_combine(self.full, self.residual, coefficients=(0.25, 0.75))

        self.assertTrue(torch.allclose(fused(), chain(), rtol=2e-6, atol=2e-6))
        self._record("mul+add", _median_ms(chain), _median_ms(fused))


if __name__ == "__main__":
    unittest.main()
