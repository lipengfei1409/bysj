"""The build keeps its noise in `build/build.log` and prints only the errors.

A hipcc build of this extension emits hundreds of warnings from torch and pybind
headers; printing them buries the two or three lines that matter, and on this
project the cost of missing the first error is another round trip to the DCU host.
"""

import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_setup_helpers():
    source = (ROOT / "setup.py").read_text()
    start = source.index("def _fallback_lines")
    end = source.index("class QuietBuildExt")
    namespace = {"re": re}
    exec(compile(source[start:end], "setup.py", "exec"), namespace)  # noqa: S102
    return namespace["_error_lines"], namespace["_fallback_lines"]


class BuildLogTests(unittest.TestCase):
    def test_only_error_lines_are_extracted(self) -> None:
        extract, _ = _load_setup_helpers()
        sample = "\n".join([
            "/src/a.cpp:12:5: error: no member named 'foo'",
            "/src/a.cpp:13:5: warning: unused variable 'x'",
            "/src/a.cpp:14:5: note: candidate function not viable",
            "ninja: build stopped: subcommand failed.",
            "/usr/bin/ld: undefined reference to `bar'",
            "FAILED: [code=1] /build/obj.o",
        ])
        lines = extract(sample)
        self.assertIn("/src/a.cpp:12:5: error: no member named 'foo'", lines)
        self.assertIn("/usr/bin/ld: undefined reference to `bar'", lines)
        self.assertIn("FAILED: [code=1] /build/obj.o", lines)
        self.assertFalse(any("warning" in line for line in lines))
        self.assertFalse(any("note:" in line for line in lines))

    def test_duplicates_collapse(self) -> None:
        # The same error repeats once per offload arch; one copy is enough.
        extract, _ = _load_setup_helpers()
        line = "/src/a.cpp:12:5: error: boom"
        self.assertEqual(extract("\n".join([line, line, line])), [line])

    def test_a_build_without_an_error_line_is_reported_as_such(self) -> None:
        extract, _ = _load_setup_helpers()
        self.assertEqual(extract("ninja: build stopped: subcommand failed."), [])

    def test_the_errors_are_read_from_the_log_not_the_exception(self) -> None:
        # The compiler writes to the redirected descriptors, so the exception text
        # is only "Error compiling objects for extension" -- scanning it finds
        # nothing, which is exactly what happened on the first narrowed build.
        source = (ROOT / "setup.py").read_text()
        self.assertIn("text = BUILD_LOG.read_text", source)
        self.assertIn("errors = _error_lines(text)", source)

    def test_a_log_without_an_error_falls_back_to_its_tail(self) -> None:
        _, fallback = _load_setup_helpers()
        sample = "\n".join([
            "/src/a.cpp:1:1: warning: unused variable",
            "/src/a.cpp:2:1: note: in instantiation of",
            "In file included from /src/a.cpp:1:",
            "ld.lld: subcommand failed",
            "ninja: build stopped: subcommand failed.",
        ])
        lines = fallback(sample)
        self.assertEqual(lines[-2:], ["ld.lld: subcommand failed",
                                      "ninja: build stopped: subcommand failed."])
        self.assertFalse(any("warning:" in line or "note:" in line for line in lines))

    def test_the_arch_list_is_split_on_every_separator(self) -> None:
        # PYTORCH_ROCM_ARCH accepts "gfx906;gfx928" and people also paste commas or
        # spaces; a multi-target list must not become one bogus target.
        source = (ROOT / "setup.py").read_text()
        self.assertIn('re.split(r"[,;\\s]+", ARCH)', source)
        splitter = __import__("re").compile(r"[,;\s]+")
        for value in ("gfx906;gfx928", "gfx906,gfx928", "gfx906 gfx928"):
            self.assertEqual([a for a in splitter.split(value) if a],
                             ["gfx906", "gfx928"], value)

    def test_the_arch_defaults_to_the_deployment_card(self) -> None:
        # gfx928 is what the model runs on; a dev box may have an older card, and
        # a build for the deployment target is the normal case.
        source = (ROOT / "setup.py").read_text()
        self.assertIn('DEFAULT_ARCH = "gfx928"', source)
        self.assertIn('ARCH = os.environ.get("PYTORCH_ROCM_ARCH") or DEFAULT_ARCH',
                      source)
        self.assertNotIn('if not _ARCH_ENV:', source)

    def test_the_vendor_header_is_patched_instead_of_the_linker(self) -> None:
        # `du_mma.h` defines host/device helpers without `inline`, so three translation
        # units including it fail the host link.  A build-local patched copy is the
        # fix -- not `--allow-multiple-definition`, which would hide real duplicates.
        source = (ROOT / "setup.py").read_text()
        start = source.index("def _patch_du_mma")
        end = source.index("PATCHED_INCLUDE = _patched_include_dir()")
        namespace = {"re": re}
        exec(compile(source[start:end], "setup.py", "exec"), namespace)  # noqa: S102
        patch = namespace["_patch_du_mma"]

        header = "\n".join([
            "  __host__ __device__ float __du_float_to_tf32(float in)",
            "  {",
            "      return in;",
            "  }",
            "",
            "  static __device__ __inline__ void already_inline() {}",
            "  __host__ __device__ inline float fine(float x) { return x; }",
            "  __host__ __device__ float another(int a) { return a; }",
        ])
        patched, count = patch(header)
        self.assertEqual(count, 2)
        self.assertIn("__host__ __device__ inline float __du_float_to_tf32", patched)
        self.assertIn("__host__ __device__ inline float another(int a)", patched)
        # an already-inline helper is left alone, not double-patched
        self.assertEqual(patched.count("inline inline"), 0)
        self.assertIn("static __device__ __inline__ void already_inline", patched)

    def test_a_header_that_needs_no_patch_is_left_alone(self) -> None:
        source = (ROOT / "setup.py").read_text()
        start = source.index("def _patch_du_mma")
        end = source.index("PATCHED_INCLUDE = _patched_include_dir()")
        namespace = {"re": re}
        exec(compile(source[start:end], "setup.py", "exec"), namespace)  # noqa: S102
        patched, count = namespace["_patch_du_mma"](
            "__host__ __device__ inline float ok(float x) { return x; }"
        )
        self.assertEqual(count, 0)
        self.assertEqual(patched, "__host__ __device__ inline float ok(float x) { return x; }")

    def test_the_arch_reaches_hipcc_as_a_flag(self) -> None:
        # The build log showed hipcc invoked with no `--offload-arch` at all: the
        # env var is advisory on some torch versions, and hipcc then compiles its
        # own default list, gfx906 included, which fails the whole unit.
        source = (ROOT / "setup.py").read_text()
        self.assertIn('ARCH_FLAGS = [f"--offload-arch={arch}" for arch in TARGETS]',
                      source)
        self.assertIn("*ARCH_FLAGS", source)
        self.assertIn("extra_compile_args={", source)

    def test_the_agent_arch_is_read_from_rocminfo(self) -> None:
        # torch's gcnArchName reported gfx906 on a gfx928 host once, which is how a
        # DUMMA build got excluded by accident; rocminfo describes the agents.
        source = (ROOT / "setup.py").read_text()
        start = source.index("def _arch_from_rocminfo")
        end = source.index("def _device_arch")
        namespace = {"re": re}
        exec(compile(source[start:end], "setup.py", "exec"), namespace)  # noqa: S102
        parse = namespace["_arch_from_rocminfo"]
        self.assertEqual(
            parse("  Name:   gfx928\n"
                  "      Name:   amdgcn-amd-amdhsa--gfx928:sramecc+:xnack-"),
            "gfx928",
        )
        self.assertEqual(parse("  Name:   gfx928\n"), "gfx928")
        self.assertEqual(parse("no gpu here"), "")

    def test_torch_and_rocminfo_disagreeing_is_reported(self) -> None:
        source = (ROOT / "setup.py").read_text()
        self.assertIn("rocminfo reports", source)
        self.assertIn("return listed or reported", source)

    def test_clean_tool_knows_the_hipified_copies(self) -> None:
        # A script rather than a `clean` command on purpose: importing setuptools'
        # vendored distutils at setup.py import time broke `setup()` itself on one
        # install ("MinimalDistribution has no attribute set_defaults").
        tool = (ROOT / "tools" / "clean_build.py").read_text()
        self.assertIn("csrc/**/*_hip.cpp", tool)
        self.assertIn("csrc/**/*_hip.h", tool)
        self.assertIn("python/kernels/_C*.so", tool)
        self.assertIn("DIRECTORIES = (\"build\",)", tool)
        self.assertNotIn("CleanBuild", (ROOT / "setup.py").read_text())
        self.assertIn("*_hip.cpp\n*_hip.h", (ROOT / ".gitignore").read_text())

    def test_the_clean_tool_lists_before_removing(self) -> None:
        # `--dry-run` is the whole reason this is safe to hand to someone on a host
        # with a slow rebuild.
        tool = (ROOT / "tools" / "clean_build.py").read_text()
        self.assertIn("--dry-run", tool)
        self.assertIn("would remove", tool)

    def test_the_engine_comparison_is_two_engines_by_default(self) -> None:
        # The default question is "how far is the ordinary kernel from the library", so
        # rocBLAS and dense; the instruction-stream engines are opt-in because they
        # measure a different design.
        tool = (ROOT / "tools" / "compare_gemm.py").read_text()
        self.assertIn('engines: list[tuple[str, object]] = [("rocBLAS"', tool)
        self.assertIn('engines.append(("dense"', tool)
        self.assertIn('if args.with_persistent:', tool)
        self.assertNotIn('engines.append(("scheduled"', tool.split("if args.with_persistent")[0])
        self.assertIn("not in this build", tool)
        self.assertIn("sys.path.insert(0, str(ROOT / \"python\"))", tool)

    def test_kernel_resources_tool_reads_the_metadata(self) -> None:
        # `hipprof --codeobj-analyze` is interactive; the same numbers are in the
        # amdgpu metadata, and occupancy is arithmetic from them.
        import importlib.util

        path = ROOT / "tools" / "kernel_resources.py"
        spec = importlib.util.spec_from_file_location("_dpa4_kernel_resources", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel = module.parse(
            "  - Name:            kern\n"
            "    NumVgprs:        112\n"
            "    NumSgprs:        40\n"
        )
        self.assertEqual(kernel[0]["Name"], "kern")
        self.assertEqual(kernel[0]["NumVgprs"], 112)
        # 65536 / (256 * 112) = 2 blocks by registers; 64 KiB / 24 KiB = 2 by LDS
        self.assertEqual(module.occupancy(112, 256, 24576), 2)
        self.assertEqual(module.occupancy(168, 256, 24576), 1)

    def test_the_bmm_attribution_tool_exists(self) -> None:
        # Shapes are known from the device profile; this answers *where* they come from,
        # which the shipped trace cannot (it has no Python stacks).
        tool = (ROOT / "tools" / "bmm_attribution.py").read_text()
        for needle in ("with_stack=True", "record_shapes=True", "deepmd/pt/",
                       "aten::bmm"):
            self.assertIn(needle, tool)

    def test_the_dense_kernel_knobs_are_forwarded(self) -> None:
        # Without a profiler on the host, the way to find what the kernel spends time
        # on is to rebuild with one decision changed -- so the switches must reach
        # hipcc, and their defaults must stay in the source.
        source = (ROOT / "setup.py").read_text()
        self.assertIn("DENSE_KNOBS", source)
        self.assertIn('("PAD", os.environ.get("DPA4_DENSE_PAD"))', source)
        self.assertIn('("KSTAGE", os.environ.get("DPA4_DENSE_KSTAGE"))', source)
        self.assertIn('("CCHUNK", os.environ.get("DPA4_DENSE_CCHUNK"))', source)
        dense = (ROOT / "csrc" / "gemm.cpp").read_text()
        self.assertIn("#ifndef DPA4_DENSE_PAD", dense)
        self.assertIn("#define DPA4_DENSE_PAD 4", dense)
        self.assertIn("constexpr int kPad = DPA4_DENSE_PAD", dense)

    def test_the_device_tests_detect_engine_support(self) -> None:
        # `csrc/` and `tests/` can be at different revisions on a host that pulls
        # nothing but the kernel: a missing `flat` keyword must downgrade the
        # measurement, not raise TypeError once per shape.
        for name in ("tests/csrc/latency/test_persistent_gemm_latency.py",
                     "tests/csrc/accuracy/test_persistent_gemm.py"):
            text = (ROOT / name).read_text()
            self.assertIn("inspect.signature(dispatch.launch_gemm)", text, name)

    def test_the_runner_can_add_the_device_suites(self) -> None:
        # One command for "everything that can run here": the device suites skip
        # themselves on a host, so the flag is safe either way.
        runner = (ROOT / "tests" / "run_python_tests.py").read_text()
        self.assertIn('"--dcu": ("tests/csrc/accuracy", "test_*.py")', runner)
        self.assertIn('"--latency": ("tests/csrc/latency", "test_*.py")', runner)
        self.assertIn("skips on a host", runner)

    def test_the_effective_arch_is_printed(self) -> None:
        # One line at the top of every build, because "which arch did this compile
        # for" was ambiguous for several rounds and is the first thing to check.
        source = (ROOT / "setup.py").read_text()
        self.assertIn('print(f"arch: building for {ARCH}"', source)
        self.assertIn('if __name__ == "__main__":', source)
        self.assertIn("EXPLICIT_ARCH = bool(os.environ.get(\"PYTORCH_ROCM_ARCH\"))",
                      source)

    def test_a_build_for_another_card_says_so(self) -> None:
        # Cross-compiling is allowed, but the binary will not run on this machine;
        # saying it here is cheaper than a HIP "no binary for GPU" later.
        source = (ROOT / "setup.py").read_text()
        self.assertIn("_device_arch()", source)
        self.assertIn("will not run here", source)

    def test_an_unsupported_arch_is_refused_unless_the_GEMM_is_scalar(self) -> None:
        source = (ROOT / "setup.py").read_text()
        self.assertIn("DUMMA_ARCHES = (", source)
        self.assertIn("does not support", source)
        self.assertIn('BUILD_SCALAR = os.environ.get("DPA4_GEMM_SCALAR") == "1"', source)

    def test_the_log_path_is_inside_build(self) -> None:
        source = (ROOT / "setup.py").read_text()
        self.assertIn('BUILD_LOG = pathlib.Path(__file__).resolve().parent / "build" / "build.log"',
                      source)
        self.assertIn('cmdclass = {"build_ext": QuietBuildExt}', source)


if __name__ == "__main__":
    unittest.main()
