"""TileLang dense FlashAttention reference, DCU parity, and opt-in benchmark.

Run the correctness tests with::

    PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
        python -m unittest discover -s tests/dcu \
        -p 'test_tilelang_flash_attention.py' -v

Add ``DPA4_RUN_FLASH_ATTN_BENCHMARK=1`` to run the synchronized benchmark on a
TileLang/HIP device. The implementation intentionally covers the dense
``flash_attn_func`` contract only: zero dropout, no GQA, windowing, ALiBi,
softcap, or returned attention probabilities.
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
import unittest

import torch
import torch.nn.functional as F

from tilelang_ops import flash_attn_func

_HAS_TILELANG_HIP = (
    importlib.util.find_spec("tilelang") is not None
    and torch.version.hip is not None
    and torch.cuda.is_available()
)
_RUN_BENCHMARK = os.environ.get("DPA4_RUN_FLASH_ATTN_BENCHMARK", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def pytorch_flash_attn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Explicit stable PyTorch reference with FlashAttention causal alignment."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "q, k, and v must have shape (batch, sequence, heads, head_dim)"
        )
    if q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            "reference requires matching batch, heads, and head dimensions"
        )
    scale = (
        1.0 / math.sqrt(q.shape[-1])
        if softmax_scale is None
        else float(softmax_scale)
    )
    q_acc = q.float().permute(0, 2, 1, 3)
    k_acc = k.float().permute(0, 2, 1, 3)
    v_acc = v.float().permute(0, 2, 1, 3)
    scores = torch.matmul(q_acc, k_acc.transpose(-1, -2)) * scale
    q_len = q.shape[1]
    kv_len = k.shape[1]
    if causal:
        q_pos = torch.arange(q_len, device=q.device)[:, None]
        kv_pos = torch.arange(kv_len, device=q.device)[None, :]
        allowed = kv_pos <= q_pos + (kv_len - q_len)
        scores = scores.masked_fill(~allowed[None, None, :, :], -torch.inf)
        valid_rows = allowed.any(dim=-1)[None, None, :, None]
        scores = torch.where(valid_rows, scores, torch.zeros_like(scores))
    else:
        valid_rows = None
    probs = torch.softmax(scores, dim=-1)
    if valid_rows is not None:
        probs = torch.where(valid_rows, probs, torch.zeros_like(probs))
    output = torch.matmul(probs, v_acc).permute(0, 2, 1, 3)
    return output.to(dtype=q.dtype)


def _open_source_flash_attn():
    try:
        from flash_attn import flash_attn_func as reference
    except Exception as error:  # optional target-specific package
        return None, f"flash_attn import failed: {type(error).__name__}: {error}"
    return reference, None


def _make_device_inputs(q_len, kv_len, dtype, *, batch=1, heads=2, head_dim=64):
    generator_device = torch.device("cuda")
    q = torch.randn(
        (batch, q_len, heads, head_dim), device=generator_device, dtype=dtype
    )
    k = torch.randn(
        (batch, kv_len, heads, head_dim), device=generator_device, dtype=dtype
    )
    v = torch.randn_like(k)
    return q, k, v


def _assert_tilelang_matches_reference(test_case, q, k, v, *, causal):
    q_tl = q.detach().clone().requires_grad_(True)
    k_tl = k.detach().clone().requires_grad_(True)
    v_tl = v.detach().clone().requires_grad_(True)
    torch.manual_seed(1729)
    grad_output = torch.randn_like(q_tl)
    output_tl = flash_attn_func(q_tl, k_tl, v_tl, causal=causal)
    grads_tl = torch.autograd.grad(output_tl, (q_tl, k_tl, v_tl), grad_output)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    output_ref = pytorch_flash_attn_reference(q_ref, k_ref, v_ref, causal=causal)
    grads_ref = torch.autograd.grad(output_ref, (q_ref, k_ref, v_ref), grad_output)

    tolerance = 8.0e-2 if q.dtype == torch.bfloat16 else 3.0e-2
    torch.testing.assert_close(output_tl, output_ref, rtol=tolerance, atol=tolerance)
    for grad_tl, grad_ref in zip(grads_tl, grads_ref):
        test_case.assertTrue(torch.isfinite(grad_tl).all().item())
        torch.testing.assert_close(grad_tl, grad_ref, rtol=tolerance, atol=tolerance)
    return output_tl, grads_tl, grad_output


class FlashAttentionReferenceTests(unittest.TestCase):
    def test_matches_torch_sdpa_and_gradients(self):
        for causal in (False, True):
            with self.subTest(causal=causal):
                torch.manual_seed(21)
                q = torch.randn((2, 5, 2, 7), dtype=torch.float64, requires_grad=True)
                k = torch.randn((2, 5, 2, 7), dtype=torch.float64, requires_grad=True)
                v = torch.randn((2, 5, 2, 7), dtype=torch.float64, requires_grad=True)
                actual = pytorch_flash_attn_reference(q, k, v, causal=causal)
                expected = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=causal,
                ).transpose(1, 2)
                torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=2.0e-6)
                upstream = torch.randn_like(actual)
                actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
                expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
                for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(
                        actual_grad, expected_grad, rtol=1.0e-5, atol=2.0e-6
                    )

    def test_bottom_right_causal_alignment_and_empty_rows(self):
        q = torch.zeros((1, 4, 1, 1), dtype=torch.float64)
        k = torch.zeros((1, 2, 1, 1), dtype=torch.float64)
        v = torch.tensor([1.0, 3.0], dtype=torch.float64).reshape(1, 2, 1, 1)
        output = pytorch_flash_attn_reference(q, k, v, causal=True)
        expected = torch.tensor([0.0, 0.0, 1.0, 2.0], dtype=torch.float64).reshape(
            1, 4, 1, 1
        )
        torch.testing.assert_close(output, expected)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_rejects_unsupported_options_before_device_dispatch(self):
        q = torch.empty((1, 2, 1, 64), dtype=torch.float16)
        with self.assertRaisesRegex(NotImplementedError, "dropout_p=0"):
            flash_attn_func(q, q, q, dropout_p=0.1)
        with self.assertRaisesRegex(NotImplementedError, "windowed"):
            flash_attn_func(q, q, q, window_size=(8, 8))
        with self.assertRaisesRegex(RuntimeError, "HIP/DCU"):
            flash_attn_func(q, q, q)


@unittest.skipUnless(_HAS_TILELANG_HIP, "requires TileLang and a HIP/DCU device")
class FlashAttentionDeviceTests(unittest.TestCase):
    def test_forward_and_backward_match_pytorch_reference(self):
        cases = ((65, 65, False), (67, 83, True))
        for q_len, kv_len, causal in cases:
            with self.subTest(q_len=q_len, kv_len=kv_len, causal=causal):
                q, k, v = _make_device_inputs(q_len, kv_len, torch.float16)
                _assert_tilelang_matches_reference(self, q, k, v, causal=causal)

    def test_bfloat16_forward_and_backward_when_supported(self):
        checker = getattr(torch.cuda, "is_bf16_supported", None)
        if checker is None:
            self.skipTest("PyTorch does not report bfloat16 support for this device")
        try:
            supported = checker()
        except Exception as error:
            self.skipTest(f"could not query device bfloat16 support: {error}")
        if not supported:
            self.skipTest("the active DCU does not support bfloat16")
        q, k, v = _make_device_inputs(65, 67, torch.bfloat16)
        _assert_tilelang_matches_reference(self, q, k, v, causal=True)

    def test_optional_open_source_flash_attn_parity(self):
        reference, reason = _open_source_flash_attn()
        if reference is None:
            self.skipTest(reason)
        q, k, v = _make_device_inputs(65, 65, torch.float16)
        output_tl, grads_tl, grad_output = _assert_tilelang_matches_reference(
            self, q, k, v, causal=True
        )
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)
        try:
            output_ref = reference(q_ref, k_ref, v_ref, dropout_p=0.0, causal=True)
            grads_ref = torch.autograd.grad(
                output_ref, (q_ref, k_ref, v_ref), grad_output
            )
        except (RuntimeError, NotImplementedError) as error:
            self.skipTest(
                "installed flash_attn does not support this HIP/DCU target: "
                f"{error}"
            )
        torch.testing.assert_close(output_tl, output_ref, rtol=3.0e-2, atol=3.0e-2)
        for grad_tl, grad_ref in zip(grads_tl, grads_ref):
            torch.testing.assert_close(grad_tl, grad_ref, rtol=5.0e-2, atol=5.0e-2)

    @unittest.skipUnless(
        _RUN_BENCHMARK,
        "set DPA4_RUN_FLASH_ATTN_BENCHMARK=1 to run the performance benchmark",
    )
    def test_opt_in_performance_benchmark(self):
        reference, reason = _open_source_flash_attn()
        q, k, v = _make_device_inputs(512, 512, torch.float16, heads=8)
        causal = False
        scale = 1.0 / math.sqrt(q.shape[-1])
        grad_output = torch.randn_like(q)
        warmup = max(1, int(os.environ.get("DPA4_FLASH_ATTN_BENCH_WARMUP", "3")))
        repeats = max(3, int(os.environ.get("DPA4_FLASH_ATTN_BENCH_REPEATS", "20")))

        def tilelang_forward():
            return flash_attn_func(q, k, v, softmax_scale=scale, causal=causal)

        def torch_forward():
            return F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=scale,
                is_causal=causal,
            ).transpose(1, 2)

        def tilelang_forward_backward():
            qg = q.detach().requires_grad_(True)
            kg = k.detach().requires_grad_(True)
            vg = v.detach().requires_grad_(True)
            output = flash_attn_func(qg, kg, vg, softmax_scale=scale, causal=causal)
            torch.autograd.grad(output, (qg, kg, vg), grad_output)

        def torch_forward_backward():
            qg = q.detach().requires_grad_(True)
            kg = k.detach().requires_grad_(True)
            vg = v.detach().requires_grad_(True)
            output = torch_forward_with_inputs(qg, kg, vg, scale, causal)
            torch.autograd.grad(output, (qg, kg, vg), grad_output)

        def torch_forward_with_inputs(qi, ki, vi, scale_value, is_causal):
            return F.scaled_dot_product_attention(
                qi.transpose(1, 2),
                ki.transpose(1, 2),
                vi.transpose(1, 2),
                scale=scale_value,
                is_causal=is_causal,
            ).transpose(1, 2)

        def measure(function):
            for _ in range(warmup):
                function()
            torch.cuda.synchronize()
            samples = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            return statistics.median(samples)

        results = {
            "TileLang forward": measure(tilelang_forward),
            "PyTorch SDPA forward": measure(torch_forward),
            "TileLang forward+backward": measure(tilelang_forward_backward),
            "PyTorch SDPA forward+backward": measure(torch_forward_backward),
        }
        flash_benchmark_error = reason
        if reference is not None:

            def flash_forward():
                return reference(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=scale,
                    causal=causal,
                )

            def flash_forward_backward():
                qg = q.detach().requires_grad_(True)
                kg = k.detach().requires_grad_(True)
                vg = v.detach().requires_grad_(True)
                output = reference(
                    qg,
                    kg,
                    vg,
                    dropout_p=0.0,
                    softmax_scale=scale,
                    causal=causal,
                )
                torch.autograd.grad(output, (qg, kg, vg), grad_output)

            try:
                results["flash_attn forward"] = measure(flash_forward)
                results["flash_attn forward+backward"] = measure(
                    flash_forward_backward
                )
                flash_benchmark_error = None
            except Exception as error:
                flash_benchmark_error = (
                    f"installed flash_attn does not support this target: {error}"
                )
        for label, milliseconds in results.items():
            print(f"{label}: {milliseconds:.4f} ms")
        for baseline, tiled in (
            ("PyTorch SDPA forward", "TileLang forward"),
            ("PyTorch SDPA forward+backward", "TileLang forward+backward"),
            ("flash_attn forward", "TileLang forward"),
            ("flash_attn forward+backward", "TileLang forward+backward"),
        ):
            if baseline in results:
                speedup = results[baseline] / results[tiled]
                print(f"{tiled} speedup vs {baseline}: {speedup:.3f}x")
        if flash_benchmark_error is not None:
            print(f"flash_attn benchmark skipped: {flash_benchmark_error}")


if __name__ == "__main__":
    unittest.main()
