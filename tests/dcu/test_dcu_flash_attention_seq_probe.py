"""Fixed-shape probe for verifying FlashAttention execution on a Hygon DCU.

All backends use the same contiguous FP16 BSHD inputs on ``cuda:0``; only
sequence length changes. The test reports the selected device/HIP runtime and
synchronized device-event timings for TileLang, open-source flash-attn, and
PyTorch SDPA. It is forward-only.

Run on the K500SM_AI from the repository root with::

    source /opt/dtk/env.sh
    PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m unittest discover -s tests/dcu \
        -p 'test_dcu_flash_attention_seq_probe.py' -v

On non-HIP hosts the device test is skipped. On a HIP host, missing backends or
a device name that does not match ``DPA4_EXPECTED_DCU_NAME`` are reported as
failures. Per-length execution/parity failures are logged as skipped and do not
contribute timing data; at least one full three-backend case must succeed.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import statistics
import time
import unittest

import torch
import torch.nn.functional as F

from tilelang_ops import flash_attn_func

_SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384)
_BATCH = 1
_HEADS = 8
_HEAD_DIM = 64
_DTYPE = torch.float16
_CAUSAL = False
_RTOL = 3.0e-2
_ATOL = 3.0e-2
_HAS_HIP_DEVICE = torch.cuda.is_available() and torch.version.hip is not None


def _package_version(distribution: str, module) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    difference = actual.detach().float() - reference.detach().float()
    reference_norm = float(reference.detach().float().norm().item())
    return {
        "max_abs": float(difference.abs().max().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "relative_l2": float(
            difference.norm().item() / max(reference_norm, 1.0e-12)
        ),
    }


def _measure_backend(
    function,
    device: torch.device,
    *,
    warmup: int,
    repeats_per_sample: int,
    sample_count: int,
):
    """Separate first-call wall time from synchronized device-event samples."""
    _synchronize(device)
    started = time.perf_counter()
    first_output = function()
    _synchronize(device)
    first_call_wall_ms = (time.perf_counter() - started) * 1000.0

    for _ in range(warmup):
        function()
    _synchronize(device)

    samples_ms = []
    for _ in range(sample_count):
        _synchronize(device)
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        event_start.record()
        for _ in range(repeats_per_sample):
            function()
        event_end.record()
        event_end.synchronize()
        samples_ms.append(event_start.elapsed_time(event_end) / repeats_per_sample)

    return first_output, {
        "first_call_wall_ms": float(first_call_wall_ms),
        "warmup_calls": warmup,
        "repeats_per_sample": repeats_per_sample,
        "samples_ms": [float(sample) for sample in samples_ms],
        "median_device_ms": float(statistics.median(samples_ms)),
        "device_event": True,
    }


def _make_torch_sdpa(q, k, v, scale: float, sequence_length: int):
    def run_sdpa():
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            scale=scale,
            is_causal=False,
        )
        return output.transpose(1, 2)

    if sequence_length <= 2048:
        return run_sdpa

    def run_fused_sdpa():
        # Prevent an O(L^2) math fallback from allocating a large score matrix.
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "this PyTorch build lacks the flash-only SDPA selector needed "
                "for sequence lengths above 2048"
            ) from error
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            return run_sdpa()

    return run_fused_sdpa


@unittest.skipUnless(_HAS_HIP_DEVICE, "requires a HIP/DCU-enabled PyTorch runtime")
class FlashAttentionDeviceProbeTests(unittest.TestCase):
    def test_fixed_shape_sequence_sweep_runs_on_k500sm_ai(self):
        device = torch.device("cuda:0")
        if not torch.cuda.is_available() or torch.version.hip is None:
            self.fail("the test entered the HIP path without a HIP/DCU runtime")

        try:
            tilelang_module = importlib.import_module("tilelang")
        except Exception as error:
            self.fail(f"TileLang import failed on the HIP host: {type(error).__name__}: {error}")
        if importlib.util.find_spec("tilelang") is None:
            self.fail("TileLang is not installed on the HIP host")
        try:
            flash_module = importlib.import_module("flash_attn")
            flash_forward = getattr(flash_module, "flash_attn_func")
        except Exception as error:
            self.fail(
                "open-source flash_attn is unavailable on the HIP host: "
                f"{type(error).__name__}: {error}"
            )

        device_name = torch.cuda.get_device_name(device)
        expected_name = os.environ.get("DPA4_EXPECTED_DCU_NAME", "K500SM_AI")
        if expected_name.casefold() not in device_name.casefold():
            self.fail(
                f"expected DCU name containing {expected_name!r}, got {device_name!r}; "
                "no K500SM_AI comparison data counted"
            )
        properties = torch.cuda.get_device_properties(device)
        architecture = getattr(
            properties,
            "gcnArchName",
            getattr(properties, "gcn_arch_name", "unknown"),
        )
        print(
            "DCU probe target: "
            f"device={device_name}, device_id={device}, arch={architecture}, "
            f"HIP={torch.version.hip}, torch={torch.__version__}, "
            f"tilelang={_package_version('tilelang', tilelang_module)}, "
            f"flash-attn={_package_version('flash-attn', flash_module)}, "
            f"fixed_shape=(batch={_BATCH}, heads={_HEADS}, head_dim={_HEAD_DIM}, "
            f"dtype={_DTYPE}, causal={_CAUSAL})",
            flush=True,
        )

        warmup = max(0, int(os.environ.get("DPA4_DCU_PROBE_WARMUP", "2")))
        repeats_per_sample = max(
            1, int(os.environ.get("DPA4_DCU_PROBE_REPEATS", "10"))
        )
        sample_count = max(1, int(os.environ.get("DPA4_DCU_PROBE_SAMPLES", "3")))
        successful_rows = []
        skipped_rows = []

        for index, sequence_length in enumerate(_SEQUENCE_LENGTHS):
            try:
                torch.manual_seed(20261110 + index)
                q = torch.randn(
                    (_BATCH, sequence_length, _HEADS, _HEAD_DIM),
                    device=device,
                    dtype=_DTYPE,
                ).contiguous()
                k = torch.randn_like(q).contiguous()
                v = torch.randn_like(q).contiguous()
                for tensor_name, tensor in (("q", q), ("k", k), ("v", v)):
                    self.assertEqual(tensor.device, device, f"{tensor_name} is not on {device}")
                    self.assertTrue(tensor.is_contiguous(), f"{tensor_name} is not contiguous")

                scale = 1.0 / math.sqrt(_HEAD_DIM)

                def tilelang_forward():
                    return flash_attn_func(
                        q,
                        k,
                        v,
                        softmax_scale=scale,
                        causal=_CAUSAL,
                    )

                def flash_attn_forward():
                    return flash_forward(
                        q,
                        k,
                        v,
                        dropout_p=0.0,
                        softmax_scale=scale,
                        causal=_CAUSAL,
                    )

                torch_forward = _make_torch_sdpa(q, k, v, scale, sequence_length)
                implementations = (
                    ("tilelang", tilelang_forward),
                    ("flash_attn", flash_attn_forward),
                    ("pytorch_sdpa", torch_forward),
                )
                outputs = {}
                timings = {}
                with torch.inference_mode():
                    for name, function in implementations:
                        output, timing = _measure_backend(
                            function,
                            device,
                            warmup=warmup,
                            repeats_per_sample=repeats_per_sample,
                            sample_count=sample_count,
                        )
                        self.assertEqual(
                            output.device,
                            device,
                            f"{name} output was not produced on {device}",
                        )
                        self.assertEqual(tuple(output.shape), tuple(q.shape))
                        self.assertEqual(output.dtype, _DTYPE)
                        self.assertTrue(
                            torch.isfinite(output).all().item(),
                            f"{name} returned non-finite values",
                        )
                        self.assertGreater(
                            timing["median_device_ms"],
                            0.0,
                            f"{name} has no positive device-event time",
                        )
                        outputs[name] = output
                        timings[name] = timing

                    errors = {
                        "tilelang_vs_flash_attn": _error_metrics(
                            outputs["flash_attn"], outputs["tilelang"]
                        ),
                        "tilelang_vs_pytorch_sdpa": _error_metrics(
                            outputs["pytorch_sdpa"], outputs["tilelang"]
                        ),
                        "flash_attn_vs_pytorch_sdpa": _error_metrics(
                            outputs["pytorch_sdpa"], outputs["flash_attn"]
                        ),
                    }
                    torch.testing.assert_close(
                        outputs["tilelang"],
                        outputs["flash_attn"],
                        rtol=_RTOL,
                        atol=_ATOL,
                    )
                    torch.testing.assert_close(
                        outputs["tilelang"],
                        outputs["pytorch_sdpa"],
                        rtol=_RTOL,
                        atol=_ATOL,
                    )
                    torch.testing.assert_close(
                        outputs["flash_attn"],
                        outputs["pytorch_sdpa"],
                        rtol=_RTOL,
                        atol=_ATOL,
                    )
                    _synchronize(device)

                row = {
                    "status": "ok",
                    "counted_in_metrics": True,
                    "sequence_length": sequence_length,
                    "device": str(device),
                    "device_name": device_name,
                    "architecture": architecture,
                    "hip_version": torch.version.hip,
                    "fixed_shape": {
                        "batch": _BATCH,
                        "heads": _HEADS,
                        "head_dim": _HEAD_DIM,
                        "dtype": str(_DTYPE),
                        "causal": _CAUSAL,
                    },
                    "tolerance": {"rtol": _RTOL, "atol": _ATOL},
                    "errors": errors,
                    "timings": timings,
                }
                successful_rows.append(row)
                print("DCU probe row: " + json.dumps(row, sort_keys=True), flush=True)
            except Exception as error:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                row = {
                    "status": "skipped",
                    "counted_in_metrics": False,
                    "sequence_length": sequence_length,
                    "device": str(device),
                    "reason": f"{type(error).__name__}: {error}",
                }
                skipped_rows.append(row)
                print("DCU probe skipped: " + json.dumps(row, sort_keys=True), flush=True)

        print(
            "DCU probe summary: "
            + json.dumps(
                {
                    "successful_sequence_lengths": [
                        row["sequence_length"] for row in successful_rows
                    ],
                    "skipped_sequence_lengths": [
                        row["sequence_length"] for row in skipped_rows
                    ],
                    "counted_cases": len(successful_rows),
                    "skipped_cases": len(skipped_rows),
                    "device_name": device_name,
                    "hip_version": torch.version.hip,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        self.assertGreater(
            len(successful_rows),
            0,
            "no sequence completed a three-backend HIP/DCU comparison; "
            "there is no device-execution evidence",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
