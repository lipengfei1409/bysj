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

import contextlib
import importlib.util
import io
import os
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
        # All twelve zoo pairs, both directions; upstream's suite walks them.
        for cf in (32, 64):
            for lmax in range(1, 7):
                for kind in ("fwd", "bwd"):
                    name = f"so2_conv_{kind}_c{cf}_l{lmax}.hip"
                    self.assertTrue((PORT / name).is_file(), name)

    def test_no_cuda_only_token_survives(self) -> None:
        offenders = hipify_tool.lint(sorted(PORT.glob("*.hip")) + sorted(PORT.glob("*.cuh")))
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

    def test_no_atens_masquerading_headers(self) -> None:
        # Regression: `ATen/hip/HIPContext.h` pulls `ATen/hip/Exceptions.h`,
        # which includes <cusolver_common.h> -- absent from every ROCm install,
        # so the generated translation could not compile.
        # The header name may still appear in a comment explaining the choice;
        # only code counts (same convention as the token lint above).
        code = "\n".join(
            line.split("//", 1)[0] for line in (PORT / "so2_conv.hip").read_text().splitlines()
        )
        self.assertNotIn("ATen/hip", code)
        glue = (PORT / "so2_conv.hip").read_text()
        self.assertIn("#include <c10/hip/HIPStream.h>", glue)
        self.assertIn("c10::hip::getCurrentHIPStream()", glue)
        # The guard must not come from torch: its header path differs between
        # builds (c10/hip/impl on one, ATen/hip/impl on another, absent on a
        # third), and the ATen one drags in cusolver.
        self.assertNotIn("HIPGuardImplMasqueradingAsCUDA", glue)
        self.assertIn("class DeviceGuard", glue)
        self.assertEqual(glue.count("dpa4::DeviceGuard device_guard"), 4)
        self.assertIn("hipGetDevice(&previous_)", glue)

    def test_dispatch_references_only_compiled_shards(self) -> None:
        # Regression: upstream's DPA4_CONV_FOR_EACH_SHAPE lists all twelve
        # (L, CF) pairs, so the dispatch referenced eleven ``conv_*_launch``
        # symbols nothing defines.  The link succeeded and ``load_library``
        # failed on the first undefined symbol.
        glue = (PORT / "so2_conv.hip").read_text()
        code = "\n".join(line.split("//", 1)[0] for line in glue.splitlines())
        self.assertNotIn("DPA4_CONV_FOR_EACH_SHAPE(DPA4_CASE)", code)
        self.assertIn("DPA4_CONV_SHIPPED_SHAPES(DPA4_CASE)", code)
        shapes = re.findall(r"macro\((\d+), (\d+)\)", code)
        self.assertTrue(shapes, "no shipped shape list found")
        import importlib.util as ilu

        spec = ilu.spec_from_file_location("so2_conv_hip_build3", PORT / "build.py")
        assert spec is not None and spec.loader is not None
        build_module = ilu.module_from_spec(spec)
        spec.loader.exec_module(build_module)
        for lmax, cf in shapes:
            for kind in ("fwd", "bwd"):
                shard = f"so2_conv_{kind}_c{cf}_l{lmax}.hip"
                self.assertIn(shard, build_module.SOURCES, f"{lmax},{cf} needs {shard}")

    def test_ridge_is_recomputed_from_hip_properties(self) -> None:
        glue = (PORT / "so2_conv.hip").read_text()
        start = glue.index("double dpa4_fp32_ridge()")
        body = glue[start : start + 900]
        self.assertIn("hipGetDeviceProperties", body)
        self.assertIn("prop.warpSize", body)
        self.assertNotIn("cudaDevAttr", body)


class BuildGuardTests(unittest.TestCase):
    def _build_module(self):
        path = PORT / "build.py"
        spec = importlib.util.spec_from_file_location("so2_conv_hip_build", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_build_refuses_a_non_hip_host(self) -> None:
        module = self._build_module()
        import torch

        if torch.version.hip is not None and torch.cuda.is_available():
            self.skipTest("this host can actually build it")
        with self.assertRaises(RuntimeError):
            module.build()
        self.assertFalse(module.registered())

    def test_dry_run_reports_the_requirement_instead_of_crashing(self) -> None:
        module = self._build_module()
        import torch

        if torch.version.hip is not None and torch.cuda.is_available():
            self.skipTest("this host can actually build it")
        buffer = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(errors):
            code = module.main(["--dry-run"])
        self.assertEqual(code, 1)
        self.assertIn("HIP operator", errors.getvalue())

    def test_declared_sources_exist(self) -> None:
        module = self._build_module()
        for name in module.SOURCES:
            self.assertTrue((PORT / name).is_file(), name)
        self.assertIn("so2_conv_fwd_c64_l3.hip", module.SOURCES)
        self.assertIn("so2_conv_bwd_c64_l3.hip", module.SOURCES)

    def test_compile_commands_shape(self) -> None:
        # torch's own include/library flags need a real torch build, so they are
        # stubbed here; the point of the test is the shape of the commands, and
        # that the sources are compiled as HIP rather than handed to ninja.
        module = self._build_module()
        module._torch_flags = lambda: (["-D_GLIBCXX_USE_CXX11_ABI=1"], ["-ltorch"])
        module.BUILD_DIR = Path("/tmp/dpa4_so2_conv_hip_test")
        commands = module.compile_commands(arch="gfx928")
        self.assertEqual(len(commands), len(module.SOURCES) + 1)
        for compile_command in commands[:-1]:
            self.assertEqual(compile_command[1], "-c")
            self.assertIn("--offload-arch=gfx928", compile_command)
            self.assertTrue(compile_command[2].endswith(".hip"))
        link = commands[-1]
        self.assertEqual(link[1], "-shared")
        self.assertIn("-ltorch", link)
        self.assertTrue(link[link.index("-o") + 1].endswith(".so"))

    def test_every_source_is_compiled_not_hipified(self) -> None:
        # Regression: torch's load() hipifies `.hip` sources and ends up with a
        # None object path, so the build must drive hipcc itself.
        module = self._build_module()
        text = (PORT / "build.py").read_text()
        self.assertIn("torch.ops.load_library", text)
        self.assertNotIn("cpp_extension.load(", text)


class RoutingTests(unittest.TestCase):
    """`model.so2_conv_hip` opens upstream's two gates and restores them."""

    def setUp(self) -> None:
        import types

        from model import so2_conv_hip

        self.module = so2_conv_hip
        self.original_env = os.environ.pop("DP_CUDA_INFER", None)
        self.guard = types.SimpleNamespace(_profitable=lambda conv: False)
        self.built = []

        def fake_build_module():
            class Stub:
                @staticmethod
                def build(*, force=False, verbose=False):
                    self.built.append((force, verbose))

                @staticmethod
                def registered():
                    return True

            return Stub

        self._real_load = so2_conv_hip._load_build_module
        self._real_guard = so2_conv_hip._guard_module
        so2_conv_hip._load_build_module = fake_build_module
        so2_conv_hip._guard_module = lambda: self.guard

    def tearDown(self) -> None:
        self.module._load_build_module = self._real_load
        self.module._guard_module = self._real_guard
        self.module.disable()
        os.environ.pop("DP_CUDA_INFER", None)
        if self.original_env is not None:
            os.environ["DP_CUDA_INFER"] = self.original_env

    def test_enable_raises_the_construction_gate_first(self) -> None:
        status = self.module.enable()
        self.assertTrue(status["enabled"])
        self.assertTrue(status["guard_overridden"])
        self.assertEqual(os.environ["DP_CUDA_INFER"], "2")
        self.assertEqual(self.built, [(False, False)])

    def test_enable_patches_the_profitability_guard(self) -> None:
        self.assertFalse(self.guard._profitable(object()))
        self.module.enable()
        self.assertTrue(self.guard._profitable(object()))

    def test_disable_restores_guard_and_environment(self) -> None:
        original = self.guard._profitable
        self.module.enable()
        self.module.disable()
        self.assertIs(self.guard._profitable, original)
        self.assertNotIn("DP_CUDA_INFER", os.environ)

    def test_enable_is_idempotent(self) -> None:
        self.module.enable()
        status = self.module.enable()
        self.assertEqual(status.get("reason"), "already enabled")
        self.assertEqual(len(self.built), 1)

    def test_a_failed_build_leaves_the_guard_alone(self) -> None:
        def failing():
            class Stub:
                @staticmethod
                def build(*, force=False, verbose=False):
                    raise RuntimeError("no visible DCU")

                @staticmethod
                def registered():
                    return False

            return Stub

        original = self.guard._profitable
        self.module._load_build_module = failing
        status = self.module.enable()
        self.assertFalse(status["enabled"])
        self.assertIn("no visible DCU", status["reason"])
        self.assertIs(self.guard._profitable, original)

    def test_status_reports_the_override(self) -> None:
        self.module.enable()
        status = self.module.status()
        self.assertTrue(status["enabled"])
        self.assertTrue(status["guard_overridden"])
        self.assertEqual(status["DP_CUDA_INFER"], "2")


class UpstreamRunnerTests(unittest.TestCase):
    """The runner that points DeepMD's own conv tests at this port."""

    def _module(self):
        path = ROOT / "tests" / "dcu" / "run_upstream_conv_test.py"
        spec = importlib.util.spec_from_file_location("run_upstream_conv_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_resolves_a_checkout_layout(self) -> None:
        module = self._module()
        with tempfile.TemporaryDirectory() as tmp:
            test = (
                Path(tmp)
                / "source"
                / "tests"
                / "pt"
                / "model"
                / "test_descriptor_sezm_cuda.py"
            )
            test.parent.mkdir(parents=True)
            test.write_text("import unittest\n")
            found = module.find_test_file(tmp, search_defaults=False)
        self.assertEqual(found, test)

    def test_missing_checkout_reports_none(self) -> None:
        module = self._module()
        self.assertIsNone(
            module.find_test_file("/nonexistent/deepmd-kit", search_defaults=False)
        )

    def test_command_filters_to_the_convolution_cases(self) -> None:
        module = self._module()
        command = module.build_argv(module.UPSTREAM_TEST, extra=["-v"])
        self.assertEqual(command[1:3], ["-m", "pytest"])
        self.assertIn("-k", command)
        self.assertEqual(command[command.index("-k") + 1], "ConvCuda")
        self.assertIn("-v", command)

    def test_child_env_makes_the_plugin_importable(self) -> None:
        module = self._module()
        env = module.child_env()
        parts = env["PYTHONPATH"].split(os.pathsep)
        self.assertIn(str(ROOT / "tests" / "dcu"), parts)
        self.assertIn(str(ROOT / "csrc" / "so2_conv_hip"), parts)

    def test_plugin_loads_the_port(self) -> None:
        import sys as _sys

        plugin = ROOT / "tests" / "dcu" / "so2_conv_hip_plugin.py"
        self.assertTrue(plugin.is_file())
        text = plugin.read_text()
        # The plugin must build (or reuse) the extension before collection, or
        # the upstream gate reads False and every case silently skips.
        self.assertIn("build.build()", text)
        self.assertIn("build.registered()", text)
        self.assertNotIn("pytest.skip", text)


if __name__ == "__main__":
    sys.exit(unittest.main())
