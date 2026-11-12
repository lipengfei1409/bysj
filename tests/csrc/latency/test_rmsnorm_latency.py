"""Opt-in latency measurements for the standalone native RMSNorm operator.

Run on a supported HIP/DTK DCU with::

    DPA4_RUN_RMSNORM_LATENCY=1 PYTHONPATH=python \
        python -m unittest discover -s tests/csrc/latency -v

Only RMSNorm is measured in this suite. The synthetic feature widths are
DPA4-relevant test points, not dimensions asserted for an external checkpoint.
"""
from __future__ import annotations

import json
import os
import statistics
import unittest

import torch

from kernels.rmsnorm import rmsnorm, rmsnorm_available


_RUN_LATENCY = os.environ.get("DPA4_RUN_RMSNORM_LATENCY", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@unittest.skipUnless(
    _RUN_LATENCY and rmsnorm_available(),
    "set DPA4_RUN_RMSNORM_LATENCY=1 and build RMSNorm on a supported DCU",
)
class RMSNormLatencyTests(unittest.TestCase):
    def test_feature_shape_latency(self):
        warmup = max(0, int(os.environ.get("DPA4_RMSNORM_WARMUP", "20")))
        iterations = max(1, int(os.environ.get("DPA4_RMSNORM_ITERATIONS", "100")))
        repeats = max(3, int(os.environ.get("DPA4_RMSNORM_REPEATS", "7")))
        shapes = ((1, 128), (32, 256), (128, 512), (512, 1024))

        with torch.inference_mode():
            for rows, width in shapes:
                with self.subTest(rows=rows, width=width):
                    x = torch.randn(rows, width, device="cuda", dtype=torch.float32)
                    scale = torch.ones(width, device="cuda", dtype=torch.float32)
                    for _ in range(warmup):
                        rmsnorm(x, scale)
                    torch.cuda.synchronize()

                    samples_ms = []
                    for _ in range(repeats):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(iterations):
                            rmsnorm(x, scale)
                        end.record()
                        end.synchronize()
                        samples_ms.append(start.elapsed_time(end) / iterations)

                    ordered = sorted(samples_ms)
                    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
                    print(
                        "RMSNorm latency: "
                        + json.dumps(
                            {
                                "rows": rows,
                                "width": width,
                                "dtype": "float32",
                                "median_us": statistics.median(samples_ms) * 1000.0,
                                "p95_us": ordered[p95_index] * 1000.0,
                                "warmup": warmup,
                                "iterations_per_repeat": iterations,
                                "repeats": repeats,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()
