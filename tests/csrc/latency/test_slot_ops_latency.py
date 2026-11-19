"""Opt-in latency comparison: slot-axis operators vs the PyTorch chain.

Run on a supported HIP/DTK DCU with::

    DPA4_RUN_SLOT_LATENCY=1 PYTHONPATH=python \\
        python -m unittest discover -s tests/csrc/latency -p 'test_slot_ops*' -v

The shapes are the ones the 3000-atom profile shows for this stage: 181838 edges,
3 -> 10 slot rows (the reserved row first), C = 64. ``slot_pack_scale`` replaces
``index_select`` + ``mul`` + ``copy_`` + ``fill_``; ``slot_reduce`` replaces the
``sum`` over the ten-row axis. Set ``DPA4_SLOT_LATENCY_JSON`` to also write the
records as a JSON array.
"""
from __future__ import annotations

import json
import os
import statistics
import unittest

import torch

from kernels.slot import slot_available, slot_pack_scale, slot_reduce

_RUN = os.environ.get("DPA4_RUN_SLOT_LATENCY", "0").lower() in {"1", "true", "yes", "on"}
_WARMUP = max(0, int(os.environ.get("DPA4_SLOT_WARMUP", "10")))
_ITERATIONS = max(1, int(os.environ.get("DPA4_SLOT_ITERATIONS", "20")))
_REPEATS = max(3, int(os.environ.get("DPA4_SLOT_REPEATS", "5")))

_EDGES = 181838
_ROWS_IN = 3
_ROWS_OUT = 10  # reserved row + nine l-rows
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
    _RUN and slot_available() and torch.cuda.is_available(),
    "set DPA4_RUN_SLOT_LATENCY=1 and build the slot HIP extension on a DCU",
)
class SlotLatencyTests(unittest.TestCase):
    records: list[dict] = []

    @classmethod
    def tearDownClass(cls):
        target = os.environ.get("DPA4_SLOT_LATENCY_JSON")
        if target:
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(cls.records, handle, indent=2, sort_keys=True)

    def _record(self, case, fused_ms, eager_ms, extra=None):
        record = {
            "case": case,
            "fused_ms": fused_ms,
            "eager_ms": eager_ms,
            "speedup": eager_ms / fused_ms if fused_ms else float("inf"),
            "edges": _EDGES,
        }
        if extra:
            record.update(extra)
        type(self).records.append(record)
        print("slot latency: " + json.dumps(record, sort_keys=True), flush=True)

    def test_pack_against_the_four_kernel_chain(self):
        slot = torch.arange(_ROWS_OUT, dtype=torch.int64, device="cuda") % _ROWS_IN
        slot[0] = -1
        src = torch.randn(_EDGES, _ROWS_IN, _CHANNELS, device="cuda")
        scale = torch.rand(_EDGES, _ROWS_OUT, _CHANNELS, device="cuda") + 0.1

        def eager():
            # index_select + mul + copy_ into a zeros buffer + fill_ the reserved row
            gathered = src.index_select(1, slot.clamp_min(0)) * scale
            out = torch.zeros(_EDGES, _ROWS_OUT, _CHANNELS, device="cuda")
            out.copy_(gathered)
            out[:, 0] = 0.0
            return out

        fused_ms = _median_ms(lambda: slot_pack_scale(src, slot, scale))
        eager_ms = _median_ms(eager)
        expected = eager()
        torch.testing.assert_close(
            slot_pack_scale(src, slot, scale), expected, rtol=2e-5, atol=2e-6
        )
        self._record("slot_pack_scale", fused_ms, eager_ms, {"channels": _CHANNELS})

    def test_reduce_against_torch_sum(self):
        src = torch.randn(_EDGES, _ROWS_OUT, _CHANNELS, device="cuda")
        weight = torch.rand(_EDGES, _ROWS_OUT, device="cuda") + 0.1
        fused_ms = _median_ms(lambda: slot_reduce(src, weight))
        eager_ms = _median_ms(lambda: (src * weight.unsqueeze(-1)).sum(dim=1))
        torch.testing.assert_close(
            slot_reduce(src, weight),
            (src * weight.unsqueeze(-1)).sum(dim=1),
            rtol=2e-4,
            atol=2e-5,
        )
        self._record("slot_reduce", fused_ms, eager_ms, {"channels": _CHANNELS})


if __name__ == "__main__":
    unittest.main()
