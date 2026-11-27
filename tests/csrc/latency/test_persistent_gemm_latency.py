"""Opt-in latency of the persistent GEMM, against rocBLAS.

Two engines, the same shapes, the same operands:

* **rocBLAS** -- the vendor SGEMM through the test-only C++ wrapper in
  `tests/dcu/rocblas_cpp.py` (the same baseline `test_triton_gemm_compare.py`
  uses), i.e. what a tuned tuned library achieves on this part;
* **persistent** -- the megakernel with a GEMM-only op pack
  (`csrc/dmk/gemm_launch.cpp`), one instruction per 16x16 output tile.

The shapes are deliberately larger than a unit test: a 256^3 product is latency
bound, and the persistent kernel's whole argument is throughput at scale.

Run on a DCU::

    MAX_JOBS=4 python setup.py build_ext --inplace
    DPA4_RUN_PERSISTENT_GEMM_LATENCY=1 PYTHONPATH=python \\
        python -m unittest tests.csrc.latency.test_persistent_gemm_latency -v

`DPA4_PERSISTENT_GEMM_JSON=<path>` writes the records; `DPA4_ROCBLAS=0` skips the
baseline (the accuracy part still runs).
"""
import inspect
import json
import os
import sys
import unittest
from pathlib import Path

import torch

from dmk import dispatch, emitters, mk
from dmk.model_types import GEMM_TILE

_DCU_TESTS = Path(__file__).resolve().parents[2] / "dcu"
if str(_DCU_TESTS) not in sys.path:
    sys.path.insert(0, str(_DCU_TESTS))

_RUN = os.environ.get("DPA4_RUN_PERSISTENT_GEMM_LATENCY", "0").lower() in {
    "1", "true", "yes", "on"
}
_WANT_ROCBLAS = os.environ.get("DPA4_ROCBLAS", "1").lower() not in {"0", "false", "no", "off"}
_WARMUP = max(0, int(os.environ.get("DPA4_PERSISTENT_GEMM_WARMUP", "5")))
_ITERATIONS = max(1, int(os.environ.get("DPA4_PERSISTENT_GEMM_ITERATIONS", "20")))
_REPEATS = max(3, int(os.environ.get("DPA4_PERSISTENT_GEMM_REPEATS", "5")))

#: (m, n, k) -- all multiples of 16.  Square up to 4096, plus rectangular shapes
#: that stress one axis (tall-skinny, wide-flat, deep-K).
SHAPES = (
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (2048, 8192, 2048),   # wide: more tiles, same K
    (8192, 2048, 2048),   # tall
    (4096, 4096, 1024),   # K-bound
    (1024, 1024, 4096),   # deep K, few tiles
)

#: CUs to fan out to: fewer than the part, the part's own count, and many.
CU_COUNTS = (4, 32, 120)

try:
    _AVAILABLE = dispatch.available()
except Exception:  # noqa: BLE001
    _AVAILABLE = False


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _time_persistent(globals_, *, iterations: int, repeats: int,
                     flat: bool = False) -> float:
    launch = globals_.instructions
    for _ in range(_WARMUP):
        dispatch.launch_gemm(launch, globals_.a, globals_.b, globals_.c, flat=flat)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            dispatch.launch_gemm(launch, globals_.a, globals_.b, globals_.c,
                                 flat=flat)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return _median(samples)


def _time_dense(a, b, *, iterations: int, repeats: int) -> float:
    for _ in range(_WARMUP):
        dispatch.launch_dense(a, b)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            dispatch.launch_dense(a, b)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return _median(samples)


def _time_callable(call, *, iterations: int, repeats: int) -> float:
    for _ in range(_WARMUP):
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
    return _median(samples)


def _supports_flat() -> bool:
    """Whether the installed `dispatch.launch_gemm` takes the `flat` switch.

    A DCU checkout can hold an older `python/dmk` than the tests (the kernel is built
    from `csrc/`, the tests come from `tests/`), and a missing keyword should downgrade
    the measurement instead of turning the run into eight identical TypeErrors.
    """
    try:
        return "flat" in inspect.signature(dispatch.launch_gemm).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic wrappers
        return False


def _tflops(m: int, n: int, k: int, milliseconds: float) -> float:
    return 2 * m * n * k / (milliseconds * 1e9) if milliseconds else 0.0


def _rocblas_baseline():
    """The vendor SGEMM callable, or None when it cannot be built here."""
    if not _WANT_ROCBLAS:
        return None, "disabled by DPA4_ROCBLAS=0"
    try:
        from rocblas_cpp import RocblasError, RocblasSgemm

        return RocblasSgemm(), None
    except (ImportError, RuntimeError) as failure:  # RocblasError is a RuntimeError
        return None, str(failure)
    except Exception as failure:  # noqa: BLE001 - report, never fail the suite
        return None, f"{type(failure).__name__}: {failure}"


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
        print(f"\n{len(cls.records)} shapes measured")

    def test_latency_against_rocblas(self) -> None:
        baseline, why = _rocblas_baseline()
        flat_available = _supports_flat() and dispatch.flat_available()
        dense_available = hasattr(dispatch, "launch_dense") and dispatch.dense_available()
        if flat_available:
            print("also measuring the role-free kernel (DPA4_GEMM_FLAT engine)")
        if baseline is None:
            print(f"\nrocBLAS baseline unavailable: {why}")

        for m, n, k in SHAPES:
            with self.subTest(shape=(m, n, k)):
                element_size = 4
                bytes_used = (m * k + k * n + m * n) * element_size
                print(f"\n{m}x{n}x{k}  ({bytes_used / 2**20:.0f} MiB of operands)")
                for cus in CU_COUNTS:
                    globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=0)
                    globals_.attach(emitters.GemmEmitter(m, n, k).queues(cus))
                    persistent_ms = _time_persistent(
                        globals_, iterations=_ITERATIONS, repeats=_REPEATS
                    )
                    dense_ms = None
                    if dense_available:
                        dense_ms = _time_dense(
                            globals_.a, globals_.b,
                            iterations=_ITERATIONS, repeats=_REPEATS,
                        )
                    flat_ms = None
                    if flat_available:
                        flat_ms = _time_persistent(
                            globals_, iterations=_ITERATIONS, repeats=_REPEATS,
                            flat=True,
                        )
                    record = {
                        "shape": [m, n, k],
                        "cus": cus,
                        "tiles": (m // GEMM_TILE) * (n // GEMM_TILE),
                        "persistent_ms": persistent_ms,
                        "persistent_tflops": _tflops(m, n, k, persistent_ms),
                    }
                    if dense_ms is not None:
                        record["dense_ms"] = dense_ms
                        record["dense_tflops"] = _tflops(m, n, k, dense_ms)
                    if flat_ms is not None:
                        record["flat_ms"] = flat_ms
                        record["flat_tflops"] = _tflops(m, n, k, flat_ms)
                        record["flat_ratio"] = (
                            persistent_ms / flat_ms if flat_ms else 0.0
                        )
                    if baseline is not None:
                        baseline_ms = _time_callable(
                            lambda: baseline(globals_.a, globals_.b),
                            iterations=_ITERATIONS,
                            repeats=_REPEATS,
                        )
                        record["rocblas_ms"] = baseline_ms
                        record["rocblas_tflops"] = _tflops(m, n, k, baseline_ms)
                        record["ratio"] = (
                            baseline_ms / persistent_ms if persistent_ms else 0.0
                        )
                        flat_note = (
                            f" | flat {record['flat_ms']:8.3f} ms "
                            f"({record['flat_tflops']:6.2f} TFLOP/s)"
                            if flat_ms is not None else ""
                        )
                        dense_note = (
                            f" | dense {record['dense_ms']:8.3f} ms "
                            f"({record['dense_tflops']:6.2f} TFLOP/s)"
                            if dense_ms is not None else ""
                        )
                        print(
                            f"  cus {cus:3d}: persistent {persistent_ms:8.3f} ms "
                            f"({record['persistent_tflops']:6.2f} TFLOP/s)"
                            f"{flat_note}{dense_note} | rocBLAS {baseline_ms:8.3f} ms "
                            f"({record['rocblas_tflops']:6.2f} TFLOP/s) | "
                            f"{(record['ratio']):5.3f}x"
                        )
                    else:
                        flat_note = (
                            f" | flat {record['flat_ms']:8.3f} ms"
                            if flat_ms is not None else ""
                        )
                        print(
                            f"  cus {cus:3d}: persistent {persistent_ms:8.3f} ms "
                            f"({record['persistent_tflops']:6.2f} TFLOP/s)"
                            f"{flat_note}"
                        )
                    type(self).records.append(record)
                    self.assertGreater(persistent_ms, 0.0)

    def test_the_persistent_product_matches_rocblas_values(self) -> None:
        # Same big operands, values compared -- timing with a wrong product is
        # not a measurement.
        for m, n, k in ((512, 512, 512), (2048, 2048, 2048)):
            with self.subTest(shape=(m, n, k)):
                globals_ = mk.GemmGlobals.from_shapes(m, n, k, seed=1)
                globals_.attach(emitters.GemmEmitter(m, n, k).queues(32))
                dispatch.launch_gemm(globals_.instructions, globals_.a, globals_.b,
                                     globals_.c)
                torch.cuda.synchronize()
                expected = globals_.a @ globals_.b
                torch.testing.assert_close(globals_.c, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
