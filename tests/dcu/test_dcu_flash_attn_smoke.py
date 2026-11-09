"""Smoke test for TileLang and flash-attn on HIP/DCU.

Run from the myproject root with::

    PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}" \
        python -m unittest discover -s tests/dcu \
        -p 'test_dcu_flash_attn_smoke.py' -v

The test skips on non-HIP hosts. On a HIP/DCU host, missing or unusable
TileLang/flash-attn installations are reported as failures rather than skips.
"""

from __future__ import annotations

import importlib
import unittest
from importlib import metadata

import torch

from tilelang_ops import flash_attn_func as tilelang_flash_attn_func

_HAS_HIP_DEVICE = torch.cuda.is_available() and torch.version.hip is not None


def _version(distribution: str, module) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return str(getattr(module, "__version__", "unknown"))


@unittest.skipUnless(_HAS_HIP_DEVICE, "requires a PyTorch HIP/DCU device")
class FlashAttentionSmokeTests(unittest.TestCase):
    def test_tilelang_and_flash_attn_forward(self):
        try:
            tilelang_module = importlib.import_module("tilelang")
        except Exception as error:
            self.fail(f"TileLang is not importable: {type(error).__name__}: {error}")

        try:
            flash_attn_module = importlib.import_module("flash_attn")
            flash_attn_func = getattr(flash_attn_module, "flash_attn_func")
        except Exception as error:
            self.fail(
                "flash-attn is not importable or does not expose flash_attn_func: "
                f"{type(error).__name__}: {error}"
            )

        device = torch.device("cuda")
        torch.manual_seed(2026)
        shape = (1, 64, 2, 64)  # batch, sequence, heads, head dimension
        q = torch.randn(shape, device=device, dtype=torch.float16)
        k = torch.randn(shape, device=device, dtype=torch.float16)
        v = torch.randn(shape, device=device, dtype=torch.float16)
        output_tl = tilelang_flash_attn_func(q, k, v, causal=False)
        output_fa = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)
        torch.cuda.synchronize(device)

        self.assertEqual(output_tl.shape, q.shape)
        self.assertEqual(output_fa.shape, q.shape)
        self.assertTrue(torch.isfinite(output_tl).all().item())
        self.assertTrue(torch.isfinite(output_fa).all().item())
        torch.testing.assert_close(output_tl, output_fa, rtol=3.0e-2, atol=3.0e-2)
        print(
            "FlashAttention smoke test passed: "
            f"device={torch.cuda.get_device_name(device)}, "
            f"torch={torch.__version__}, HIP={torch.version.hip}, "
            f"tilelang={_version('tilelang', tilelang_module)}, "
            f"flash-attn={_version('flash-attn', flash_attn_module)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
