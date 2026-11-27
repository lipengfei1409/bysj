"""Opt-in latency measurement for the persistent GEMM.

Records milliseconds per launch and the achieved TFLOP/s for a few shapes, so a
change to `ops/gemm.h` or to the per-CU split can be compared against a number
instead of an impression.

Run on a DCU with::

    DPA4_RUN_PERSISTENT_GEMM_LATENCY=1 PYTHONPATH=python \\
        python -m unittest discover -s tests/csrc/latency -p 'test_persistent_gemm*' -v

``DPA4_PERSISTENT_GEMM_JSON`` writes the records as a JSON array.
"""
import json
import os
import unittest

import torch

from dmk import dispatch, emitters, mk
from dmk.model_types import GEMM_TILE

_RUN = os.environ.get("DPA4_RUN_PERSISTENT_GEMM_LATENCY", "0").lower() in {
    "1", "true", "yes", "on"
}
_ITERATIONS = max(1, int(os.environ.get("DPA4_PERSISTENT_GEMM_ITERATIONS", "20")))

#: (m, n, k, cus) -- small enough to fit, wide enough to show the trend.
CASES = (
    (256, 256, 256, 4),
    (1024, 1024, 1024, 32),
    (2048, 2048, 2048, 64),
)

try:
    _AVAILABLE = dispatch.available()
except Exception:  # noqa: BLE001
    _AVAILABLE = False


@unittest.skipUnless(
    _RUN and _AVAILABLE and torch.cuda.is_available(),
    "set DPA4_RUN_PERSISTENT_GEMM_LATENCY=1, build the extension and use a DCU",
)
class PersistentGemmLatenyTests(unittest.TestCase):
    records: list = []

    @classmethod
    def tearDownClass(cls) -> None:
        path = os.environ.get("DPA4_PERSISTENT_GEMM_JSON")
        if path and cls.records:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(cls.records, handle, indent=2)
            print(f"\npersistent GEMM records: {path}")

    def test_latency_by_shape(self) -> None:
        for m, n, k, cus in CASES:
            with self.subTest(shape=(m, n, k), cus=cus):
                globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=0)
                globals_.attach(emitters.GemmEmitter(m, n, k).queues(cus))
                milliseconds = mk.time_gemm(globals_, iterations=_ITERATIONS)
                record = {
                    "shape": [m, n, k],
                    "cus": cus,
                    "tiles": (m // GEMM_TILE) * (n // GEMM_TILE),
                    "ms": milliseconds,
                    "tflops": 2 * m * n * k / (milliseconds * 1e9) if milliseconds else 0.0,
                }
                type(self).records.append(record)
                print("\n" + json.dumps(record))
                self.assertGreater(milliseconds, 0.0)


if __name__ == "__main__":
    unittest.main()
