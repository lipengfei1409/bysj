"""Opt-in latency comparison: fused block-diagonal GEMM vs slice-and-bmm.

The dense form slices each diagonal block out of the strided weight, issues one
``torch.bmm`` per block and concatenates.  The fused kernel consumes the blocks
as stored and writes the output once.

Run on a DCU with::

    DPA4_RUN_BLOCKDIAG_LATENCY=1 PYTHONPATH=python \\
        python -m unittest discover -s tests/csrc/latency -p 'test_blockdiag*' -v
"""
import json
import os
import statistics
import unittest

import torch

from kernels.blockdiag import blockdiag_available, blockdiag_gemm

_RUN = os.environ.get("DPA4_RUN_BLOCKDIAG_LATENCY", "0").lower() in {"1", "true", "yes", "on"}
_WARMUP = max(0, int(os.environ.get("DPA4_BLOCKDIAG_WARMUP", "10")))
_ITERATIONS = max(1, int(os.environ.get("DPA4_BLOCKDIAG_ITERATIONS", "20")))
_REPEATS = max(3, int(os.environ.get("DPA4_BLOCKDIAG_REPEATS", "5")))

_EDGES = 181838
_SIZES = (256, 384)  # M0 and M1 of the profiled mixing layer


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
    _RUN and blockdiag_available() and torch.cuda.is_available(),
    "set DPA4_RUN_BLOCKDIAG_LATENCY=1 and build the HIP extension on a DCU",
)
class BlockDiagonalLatencyTests(unittest.TestCase):
    records: list = []

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get("DPA4_BLOCKDIAG_LATENCY_JSON")
        if path and cls.records:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(cls.records, handle, indent=2)
            print(f"\nblockdiag latency records: {path}")

    def test_gemm_against_slice_and_bmm(self):
        torch.manual_seed(0)
        weights = torch.stack([torch.randn(s, s) for s in _SIZES]).to("cuda")
        offsets = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(_SIZES), 0).tolist()),
            dtype=torch.int64,
            device="cuda",
        )
        rows = int(offsets[-1])
        x = torch.randn(_EDGES, rows, device="cuda")

        def dense():
            parts = []
            for b, size in enumerate(_SIZES):
                start = int(offsets[b])
                parts.append(x[:, start : start + size] @ weights[b].t())
            return torch.cat(parts, dim=1)

        fused_ms = _median_ms(lambda: blockdiag_gemm(x, weights, offsets))
        dense_ms = _median_ms(dense)
        record = {
            "op": "blockdiag_gemm",
            "edges": _EDGES,
            "rows": rows,
            "blocks": list(_SIZES),
            "fused_ms": fused_ms,
            "dense_ms": dense_ms,
            "speedup": dense_ms / fused_ms if fused_ms else 0.0,
        }
        self.records.append(record)
        print("\n" + json.dumps(record))
        self.assertGreater(fused_ms, 0.0)
        self.assertGreater(dense_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
