"""Host tests for the translated HIP fused SO(2) convolution.

Nothing here compiles HIP: the port is checked structurally, which is exactly
what can go wrong silently.  The two traps worth pinning are

* a CUDA-only token surviving the translation (it would only fail on the DCU,
  after a multi-minute hipcc run), and
* the 32-lane unit being "fixed" to 64 while supposedly porting.  The kernel
  indexes every output column as ``j * 32 + lane``; widening the shuffle to a
  full wavefront would change the algorithm, not translate it, so the shuffle
  width must stay ``kWarp``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PORT = ROOT / "csrc" / "so2_conv_hip"
TOOL = PORT / "tools" / "hipify_so2_conv.py"

_spec = importlib.util.spec_from_file_location("hipify_so2_conv", TOOL)
assert _spec is not None and _spec.loader is not None
hipify_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hipify_tool)

TRANSLATED = (
    "so2_conv.hip",
    "so2_conv_hip.cuh",
    "so2_conv_kernel_hip.cuh",
    "so2_conv_launch_hip.h",
    "so2_conv_instantiate_hip.cuh",
    "so2_conv_fwd_c64_l3.hip",
    "so2_conv_bwd_c64_l3.hip",
)


def _schema(glue: str, name: str) -> str:
    """Return one ``m.def`` schema with its C string literals joined.

    The source splits the schema across adjacent literals for line length, so a
    raw substring match would fail on formatting rather than on content.
    """
    start = glue.index(f"m.def(")
    while True:
        start = glue.index(name + "(", start)
        head = glue.rfind("m.def(", 0, start)
        if head != -1:
            break
    end = glue.index(");", start)
    literals = re.findall(r'"([^"]*)"', glue[head:end])
    return " ".join("".join(literals).split())


class TranslationTests(unittest.TestCase):
    def test_every_translated_file_is_present(self) -> None:
        for name in TRANSLATED:
            self.assertTrue((PORT / name).is_file(), name)

    def test_no_cuda_only_token_survives(self) -> None:
        offenders = hipify_tool.lint([PORT / name for name in TRANSLATED])
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_lint_catches_a_planted_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.hip"
            path.write_text("cudaStream_t stream;\n// cudaStream_t in a comment is fine\n")
            offenders = hipify_tool.lint([path])
        self.assertEqual(len(offenders), 1, offenders)
        self.assertIn("cudaStream_t", offenders[0])

    def test_shuffle_width_stays_at_the_32_lane_unit(self) -> None:
        header = (PORT / "so2_conv_hip.cuh").read_text()
        self.assertIn("__shfl_xor(v, d, kWarp)", header)
        self.assertIn("constexpr int kWarp = 32;", header)
        # A width of warpSize would silently change the lane mapping on a
        # 64-lane wavefront.
        self.assertNotIn("__shfl_xor(v, d, warpSize)", header)

    def test_divergent_barrier_uses_the_wave_builtin(self) -> None:
        compat = (PORT / "hip_compat.cuh").read_text()
        self.assertIn("__builtin_amdgcn_wave_barrier", compat)
        kernel = (PORT / "so2_conv_kernel_hip.cuh").read_text()
        self.assertNotIn("__syncthreads();\n      __syncwarp", kernel)
        self.assertIn("dpa4_wave_barrier()", kernel)
        # No CUDA warp-level primitive may remain in the kernel body.
        for line in kernel.splitlines():
            code = line.split("//", 1)[0]
            self.assertNotIn("__syncwarp", code, line)

    def test_async_pipeline_is_synchronous_and_marked(self) -> None:
        kernel = (PORT / "so2_conv_kernel_hip.cuh").read_text()
        self.assertIn("__builtin_memcpy(into + at, from + at", kernel)
        self.assertIn("v1: synchronous stage", kernel)
        self.assertNotIn("__pipeline", kernel)

    def test_the_registry_reuses_the_deepmd_namespace(self) -> None:
        glue = (PORT / "so2_conv.hip").read_text()
        # DeepMD's own Python wrapper calls torch.ops.deepmd.dpa4_so2_conv, so
        # registering into its namespace is what avoids a source edit upstream.
        self.assertIn("TORCH_LIBRARY_FRAGMENT(deepmd, m)", glue)
        self.assertIn("dpa4_so2_conv(", _schema(glue, "dpa4_so2_conv"))
        self.assertIn('m.impl("dpa4_so2_conv", torch::kCUDA', glue)

    def test_schema_matches_upstream(self) -> None:
        glue = (PORT / "so2_conv.hip").read_text()
        expected = (
            "dpa4_so2_conv(Tensor x, Tensor src, Tensor dst, Tensor dst_order, "
            "Tensor dst_rowptr, Tensor src_order, Tensor src_rowptr, Tensor runs, "
            "Tensor kc, Tensor cb, Tensor w0, Tensor w1, Tensor gw, Tensor q, "
            "Tensor k, Tensor logit_w, Tensor null_logit, Tensor env, Tensor rad0, "
            "Tensor fscale, Tensor head_gate, Tensor rescale, "
            "int lmax, int focus_dim, int rank) "
            "-> (Tensor out, Tensor alpha, Tensor pre_gate, Tensor z_all)"
        )
        self.assertEqual(_schema(glue, "dpa4_so2_conv"), " ".join(expected.split()))
        backward = (
            "dpa4_so2_conv_backward(Tensor grad_out, Tensor z_all, Tensor x, "
            "Tensor src, Tensor dst, Tensor src_order, Tensor src_rowptr, "
            "Tensor runs, Tensor kc, Tensor cb, Tensor w0, Tensor w1, Tensor gw, "
            "Tensor alpha, Tensor head_gate, Tensor rescale, "
            "int lmax, int focus_dim, int rank) "
            "-> (Tensor g_x, Tensor g_runs, Tensor g_kc, Tensor g_alpha)"
        )
        self.assertEqual(
            _schema(glue, "dpa4_so2_conv_backward"), " ".join(backward.split())
        )

    def test_ridge_is_recomputed_from_hip_properties(self) -> None:
        glue = (PORT / "so2_conv.hip").read_text()
        start = glue.index("double dpa4_fp32_ridge()")
        body = glue[start : start + 900]
        self.assertIn("hipGetDeviceProperties", body)
        self.assertIn("prop.warpSize", body)
        self.assertNotIn("cudaDevAttr", body)


class BuildGuardTests(unittest.TestCase):
    def test_build_refuses_a_non_hip_host(self) -> None:
        spec = importlib.util.spec_from_file_location("so2_conv_hip_build", PORT / "build.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import torch

        if torch.version.hip is not None and torch.cuda.is_available():
            self.skipTest("this host can actually build it")
        with self.assertRaises(RuntimeError):
            module.build()
        self.assertFalse(module.registered())

    def test_declared_sources_exist(self) -> None:
        spec = importlib.util.spec_from_file_location("so2_conv_hip_build2", PORT / "build.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in module.SOURCES:
            self.assertTrue((PORT / name).is_file(), name)
        self.assertIn("so2_conv_fwd_c64_l3.hip", module.SOURCES)
        self.assertIn("so2_conv_bwd_c64_l3.hip", module.SOURCES)


if __name__ == "__main__":
    sys.exit(unittest.main())
