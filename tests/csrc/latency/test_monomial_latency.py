"""Opt-in latency comparison: fused monomial runs vs the dense chain.

The dense chain is what the profile shows: a power table over the quaternion
components, a gather/product to form the design matrix, and one matmul against
the run coefficients.  The fused kernel evaluates the monomials in registers and
contracts in the same pass, so the comparison is "three materialised passes vs
one".

Run on a DCU with::

    DPA4_RUN_MONOMIAL_LATENCY=1 PYTHONPATH=python \\
        python -m unittest discover -s tests/csrc/latency -p 'test_monomial*' -v

Set ``DPA4_MONOMIAL_LATENCY_JSON`` to also write the records as a JSON array.
"""
import json
import os
import statistics
import unittest

import torch

from kernels.monomial import monomial_runs, monomial_runs_available

_RUN = os.environ.get("DPA4_RUN_MONOMIAL_LATENCY", "0").lower() in {"1", "true", "yes", "on"}
_WARMUP = max(0, int(os.environ.get("DPA4_MONOMIAL_WARMUP", "10")))
_ITERATIONS = max(1, int(os.environ.get("DPA4_MONOMIAL_ITERATIONS", "20")))
_REPEATS = max(3, int(os.environ.get("DPA4_MONOMIAL_REPEATS", "5")))

_EDGES = 181838  # the profiled 3000-atom box
_DEGREE = 6  # lmax = 3 needs monomials up to 2 * lmax
_RUNS = 46  # NW = 3 * (lmax + 1) ** 2 - 2


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


def _tables(degree):
    exps = []
    for total in range(degree + 1):
        for a in range(total + 1):
            for b in range(total - a + 1):
                for c in range(total - a - b + 1):
                    exps.append([a, b, c, total - a - b - c])
    return torch.tensor(exps, dtype=torch.int8, device="cuda")


def _dense(q, exps, coeff):
    """Power table -> design matrix -> matmul, the chain the kernel replaces."""
    powers = torch.stack(
        [torch.pow(q[:, i : i + 1], torch.arange(int(exps[:, i].max()) + 1, device="cuda"))
         for i in range(4)]
    )  # (4, E, P)
    design = torch.ones(q.shape[0], exps.shape[0], device="cuda")
    for m in range(exps.shape[0]):
        for i in range(4):
            design[:, m] *= powers[i][:, int(exps[m, i])]
    return design @ coeff.t()


@unittest.skipUnless(
    _RUN and monomial_runs_available() and torch.cuda.is_available(),
    "set DPA4_RUN_MONOMIAL_LATENCY=1 and build the HIP extension on a DCU",
)
class MonomialLatencyTests(unittest.TestCase):
    records: list = []

    @classmethod
    def tearDownClass(cls):
        path = os.environ.get("DPA4_MONOMIAL_LATENCY_JSON")
        if path and cls.records:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(cls.records, handle, indent=2)
            print(f"\nmonomial latency records: {path}")

    def test_runs_against_the_dense_chain(self):
        torch.manual_seed(0)
        q = torch.randn(_EDGES, 4, device="cuda")
        q = (q / q.norm(dim=1, keepdim=True)).contiguous()
        exps = _tables(_DEGREE)
        coeff = torch.randn(_RUNS, exps.shape[0], device="cuda")

        fused_ms = _median_ms(lambda: monomial_runs(q, exps, coeff))
        dense_ms = _median_ms(lambda: _dense(q, exps, coeff))
        record = {
            "op": "monomial_runs",
            "edges": _EDGES,
            "monomials": int(exps.shape[0]),
            "runs": _RUNS,
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
