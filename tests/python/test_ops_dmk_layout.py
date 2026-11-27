"""Structural tests for the ops/ + dmk/ split.

``csrc/ops/`` owns the operator templates and the factory; ``csrc/dmk/`` owns
what persistent scheduling needs.  The split is only worth anything if it holds:
an op must not reach into the runtime, the runtime must not own op templates, and
every pair that shares arithmetic must share it through a header rather than by
duplicating it.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "csrc" / "ops"
DMK = ROOT / "csrc" / "dmk"


def _headers(directory: Path) -> list[Path]:
    """Source headers of a directory, excluding torch's hipified copies.

    hipify writes `<name>_hip.h` / `<name>_hip.cpp` next to the sources before
    compiling them, so a built tree contains build output that looks like source.  It
    is in `.gitignore`, and these checks are about the files we write.
    """
    return sorted(p for p in directory.glob("*.h") if not p.name.endswith("_hip.h"))


class NamespaceTests(unittest.TestCase):
    """Which namespace each shared name lives in -- the mistake that cost two builds."""

    def test_the_launcher_qualifies_every_shared_name(self) -> None:
        launcher = (DMK / "megakernel.cpp").read_text()
        # `tensor` and `config` are aliases inside dmk::mk, not members of dmk;
        # kLaunchBytes is in ops, not in ops::config.
        for wrong in ("dmk::tensor::", "dmk::config::", "dmk::mk::config::kLaunchBytes"):
            self.assertNotIn(wrong, launcher, wrong)
        self.assertIn("ops::tensor::from_torch", launcher)
        self.assertIn("ops::config::kInstructionWidth", launcher)
        # the launch mechanics moved to dmk/launch.h, which is where the LDS
        # request and the grid size now live
        shared = (DMK / "launch.h").read_text()
        self.assertIn("config::kThreads", shared)
        self.assertIn("kLaunchBytes", shared)

    def test_k_launch_bytes_is_defined_in_ops_not_config(self) -> None:
        config = (OPS / "config.h").read_text()
        marker = config.index("kLaunchBytes =")
        before = config[:marker]
        # the nearest preceding namespace close tells which namespace we are in
        self.assertGreater(before.rindex("}  // namespace ops::config"),
                           before.rfind("namespace ops::config {"),
                           "kLaunchBytes unexpectedly moved inside ops::config")
        self.assertIn("using ops::kLaunchBytes;", (DMK / "megakernel.h").read_text())


class TwoPhaseLookupTests(unittest.TestCase):
    """Names used inside dmk templates must be visible at the definition site."""

    def test_every_ops_name_used_in_the_runtime_is_imported(self) -> None:
        # `make_op<Globals, Ops...>` is a template-id with explicit arguments: it
        # resolves at the definition site and never through ADL, so a missing
        # `using` is a hard error rather than a fallback.
        for header in ("state.h", "controller.h"):
            text = (DMK / header).read_text()
            for name in ("make_op", "BaseOp", "InstructionState"):
                if f"{name}<" in text:
                    self.assertIn(f"using ops::{name};", text, f"{header}: {name}")

    def test_the_runtime_does_not_use_unqualified_op_names(self) -> None:
        # Everything in ops/ is reached through a `using` or a qualified name.
        for header in ("state.h", "controller.h", "megakernel.h", "launch.h"):
            text = (DMK / header).read_text()
            self.assertNotIn("dmk::tensor::", text)
            self.assertNotIn("dmk::config::", text)


class StandaloneKernelNamespaceTests(unittest.TestCase):
    """A kernel outside `ops/` must import the names it borrows."""

    def test_the_dense_kernel_aliases_config_and_dumma(self) -> None:
        # csrc/gemm.cpp is deliberately not an operator, so it sits outside the
        # namespace that declares config/dumma and has to alias them itself.  The
        # first build failed on exactly this.
        dense = (ROOT / "csrc" / "gemm.cpp").read_text()
        self.assertIn("namespace config = ops::config;", dense)
        self.assertIn("namespace dumma = ops::dumma;", dense)
        self.assertIn("::du::dumma::DUFragment", dense)

    def test_the_flat_kernel_inherits_the_runtime_aliases(self) -> None:
        # It lives in dmk::mk, where megakernel.h already aliases config/lds/tensor.
        flat = (DMK / "gemm_flat.cpp").read_text()
        self.assertIn("namespace dmk::mk {", flat)
        # config comes from megakernel.h; the fragment API does not, because only a
        # kernel uses it -- and using it without the alias was this file's first build
        self.assertNotIn("namespace config =", flat)
        self.assertIn("namespace dumma = ops::dumma;", flat)


class LdsStrideTests(unittest.TestCase):
    """The dense kernel's LDS rows are padded, and why that is not a swizzle."""

    def test_the_stage_strides_are_padded(self) -> None:
        dense = (ROOT / "csrc" / "gemm.cpp").read_text()
        self.assertIn("constexpr int kStrideA = kKT + kPad", dense)
        self.assertIn("constexpr int kStrideB = kNT + kPad", dense)
        # the fragment loader is told the padded stride, which is the whole point
        self.assertIn("static_cast<unsigned>(kStrideA)", dense)
        self.assertIn("static_cast<unsigned>(kStrideB)", dense)

    def test_the_padding_is_explained_as_a_loader_constraint(self) -> None:
        # A swizzled stage would disagree with `ld_matrix`, which only takes a leading
        # dimension -- that must be written down or the next reader will "fix" it.
        dense = (ROOT / "csrc" / "gemm.cpp").read_text()
        self.assertIn("du_load_matrix_sync", dense)
        self.assertIn("plain row-major", dense)
        self.assertIn("ShmemSwizzleLayout", dense)


class BuildIncludePathTests(unittest.TestCase):
    """A cwd-relative include dir breaks every header include from a subdirectory."""

    def test_the_csrc_include_dir_is_absolute(self) -> None:
        # torch's ninja backend runs the compiler from the build temp directory,
        # so `-Icsrc` points at nothing there and `csrc/dmk/*.cpp` cannot see
        # `ops.h` (or `ops/...` from any header).  It must be absolute.
        setup_py = (ROOT / "setup.py").read_text()
        self.assertIn("str(CSRC)]", setup_py)
        self.assertIn("CSRC = pathlib.Path(__file__).resolve().parent", setup_py)
        self.assertNotIn('include_dirs=["csrc"]', setup_py)

    def test_the_launcher_reaches_the_operator_headers(self) -> None:
        launcher = (DMK / "megakernel.cpp").read_text()
        for used in ('#include "ops/gemm.h"', '#include "ops/noop.h"'):
            self.assertIn(used, launcher)
        # `#include "ops.h"` resolves only through the include dir, which is the
        # path the build above guarantees.
        self.assertIn('#include "checks.h"', launcher)
        # the launch itself is shared, not re-implemented here
        self.assertIn('#include "dmk/launch.h"', launcher)
        self.assertNotIn('hipFuncSetAttribute', launcher)
        self.assertIn("launch_kernel(mk_all, g, blocks", launcher)
        self.assertIn("bool launch(const MkGlobals& g", launcher)


class GemmWaveContractTests(unittest.TestCase):
    """The host and the kernel must agree on how many tiles one instruction covers."""

    def test_the_block_shape(self) -> None:
        header = (OPS / "config.h").read_text()
        consumers = int(re.search(r"kConsumerWaves\s*=\s*(\d+)", header).group(1))
        service = int(re.search(r"kServiceWaves\s*=\s*(\d+)", header).group(1))
        wave = int(re.search(r"kWaveSize\s*=\s*(\d+)", header).group(1))
        self.assertEqual(consumers, 1)
        self.assertEqual((consumers + service) * wave, 256)
        # one C page per consumer wave, plus the A and B pages
        pages = int(re.search(r"kNumPages\s*=\s*(\d+)", header).group(1))
        self.assertGreaterEqual(pages, 2 + consumers)

    def test_the_tile_run_length_matches_the_consumer_wave_count(self) -> None:
        from dmk.model_types import GEMM_TILES_PER_INSTRUCTION

        header = (OPS / "config.h").read_text()
        match = re.search(r"kConsumerWaves\s*=\s*(\d+)", header)
        self.assertIsNotNone(match, "config.h no longer defines kConsumerWaves")
        self.assertEqual(int(match.group(1)), GEMM_TILES_PER_INSTRUCTION,
                         "host GEMM_TILES_PER_INSTRUCTION drifted from config.h")

    def test_the_kernel_stages_one_slice_per_wave(self) -> None:
        gemm = (OPS / "gemm.h").read_text()
        self.assertIn("kTilesPerInstruction = config::kConsumerWaves", gemm)
        self.assertIn("wave * kSliceElements", gemm)
        # every consumer wave computes, so run() must not gate on `wave == 0`
        self.assertNotIn("if (wave == 0) {", gemm.split("__device__ void store")[0])

    def test_the_kernel_declares_its_block_size(self) -> None:
        # A kernel compiled without `__launch_bounds__` gets hipcc's default, and the
        # three service waves beyond it are silently dropped at launch: correct for a
        # single-tile shape, wrong for everything else.
        kernel = (DMK / "megakernel.cpp").read_text()
        self.assertIn("__launch_bounds__(config::kThreads) mk_all", kernel)
        # the role-free kernel declares its own block: it is not the scheduler's
        flat = (DMK / "gemm_flat.cpp").read_text()
        self.assertIn("__launch_bounds__(kFlatThreads)", flat)
        self.assertNotIn("config::kThreads", flat)
        self.assertIn("kFlatWaves = 4", flat)
        # ...and a clamped launch is refused instead of run: the driver would
        # otherwise drop the service waves without a word.
        shared = (DMK / "launch.h").read_text()
        self.assertIn("maxThreadsPerBlock", shared)
        self.assertIn("__launch_bounds__(<block size>) to the kernel", shared)
        self.assertIn("uint32_t threads = config::kThreads", shared)
        self.assertIn("kThreads = kTotalWaves * kWaveSize", (OPS / "config.h").read_text())

    def test_the_k_loop_is_unrolled(self) -> None:
        gemm = (OPS / "gemm.h").read_text()
        self.assertIn("#pragma unroll", gemm.split("__device__ void run")[1])


class LayoutTests(unittest.TestCase):
    def test_ops_holds_the_templates_and_the_factory(self) -> None:
        names = {p.name for p in _headers(OPS)}
        for expected in ("op.h", "config.h", "elementwise.h", "gemm.h", "tile.h",
                         "lds.h", "state.h"):
            self.assertIn(expected, names)

    def test_dmk_holds_only_the_persistent_runtime(self) -> None:
        names = {p.name for p in _headers(DMK)}
        self.assertEqual(names,
                         {"megakernel.h", "state.h", "controller.h", "launch.h"})
        self.assertTrue((DMK / "megakernel.cpp").is_file())

    def test_ops_never_include_anything_under_dmk(self) -> None:
        # The rule, no exceptions: an operator header may not depend on the
        # runtime.  The shared-memory contract it *is* written against lives in
        # ops/state.h, so `ops` stays self-contained.
        for header in _headers(OPS):
            text = header.read_text()
            self.assertNotIn('#include "dmk/', text, header.name)

    def test_the_runtime_may_include_ops(self) -> None:
        # The VM drives ops, so the dependency points this way only.
        text = (DMK / "megakernel.h").read_text()
        self.assertIn('#include "ops/', text)

    def test_ops_live_in_the_ops_namespace(self) -> None:
        # Everything under csrc/ops/ is declared in ops:: -- the operator
        # templates, the config contract, the LDS and tensor helpers.  The one
        # exception is the forward declaration of the runtime's per-wave State,
        # which has to name its real home (dmk::mk).
        for header in _headers(OPS):
            text = header.read_text()
            # No dmk:: namespace may be opened here, not even for the State
            # forward declaration: that lives in ops/state.h now.
            self.assertEqual(
                re.findall(r"\nnamespace (dmk::[A-Za-z_:]+) \{", text), [], header.name
            )

    def test_the_state_contract_lives_with_the_ops(self) -> None:
        # `State` is the interface between runtime and operator, so it is
        # declared in ops/state.h and hooks take a bare `State&`.
        template = (OPS / "op.h").read_text()
        self.assertIn("struct State;", template)
        self.assertNotIn("dmk::", template)
        self.assertIn("State& state", template)
        contract = (OPS / "state.h").read_text()
        self.assertIn("struct State {", contract)
        for header in _headers(OPS):
            self.assertNotIn("dmk::", header.read_text(), header.name)

    def test_the_runtime_aliases_the_ops_namespace(self) -> None:
        for path in (DMK / "state.h", DMK / "megakernel.h", DMK / "controller.h"):
            text = path.read_text()
            self.assertIn("namespace config = ops::config;", text, path.name)
            self.assertIn('namespace dmk::mk {', text, path.name)

    def test_the_runtime_builds_on_the_ops_contract(self) -> None:
        text = (DMK / "state.h").read_text()
        self.assertIn('#include "ops/state.h"', text)
        self.assertIn("using ops::State;", text)

    def test_includes_name_their_folder(self) -> None:
        # A bare `#include "op.h"` would resolve only by luck once the files are
        # in two directories; every cross-folder include is spelled out.
        known = {p.name for p in _headers(OPS)} | {p.name for p in _headers(DMK)}
        for directory in (OPS, DMK):
            for path in [*_headers(directory),
                         *(p for p in sorted(directory.glob("*.cpp"))
                           if not p.name.endswith("_hip.cpp"))]:
                for match in re.finditer(r'#include "([A-Za-z0-9_./-]+)"', path.read_text()):
                    target = match.group(1)
                    if "/" not in target:
                        self.assertNotIn(target, known, f"{path.name}: bare {target}")
                    else:
                        self.assertTrue((OPS.parent / target).is_file(), f"{path.name}: {target}")


ORDINARY = ROOT / "csrc" / "so2_conv.cpp"


class SpecificOptimisationTests(unittest.TestCase):
    """`csrc/ops/` is for generic persistent ops; specific work is in `csrc/`."""

    def test_the_specific_optimisations_live_in_csrc(self) -> None:
        self.assertTrue((ROOT / "csrc" / "so2_conv.cpp").is_file())
        self.assertTrue((ROOT / "csrc" / "so2_conv_math.h").is_file())
        self.assertTrue((ROOT / "csrc" / "gie.cpp").is_file())  # GIE / zonal scatter
        for gone in ("so2_conv_op.h", "so2_conv_math.h", "mode.h"):
            self.assertFalse((OPS / gone).exists(), gone)

    def test_ops_holds_only_the_generic_operators(self) -> None:
        ops = {p.name for p in _headers(OPS)}
        self.assertEqual(
            ops,
            {
                "op.h", "state.h", "config.h", "lds.h", "copy.h", "tensor.h",
                "tile.h", "dtype.h", "dumma.h", "utils.h",
                "noop.h", "elementwise.h", "embedding.h", "gemm.h", "attention.h",
            },
        )

    def test_gemm_has_one_class_and_runs_on_dumma(self) -> None:
        # One class, and the arithmetic is the matrix unit -- a second standalone
        # GEMM would be two implementations again, whichever one it wraps.
        gemm = (OPS / "gemm.h").read_text()
        self.assertIn("class GemmOp final : public BaseOp<Globals>", gemm)
        self.assertEqual(gemm.count("class "), 1)
        for gone in ("GemmWaveT", "GemmSharedT", "GemmParamsT", "run_gemm_wave"):
            self.assertNotIn(gone, gemm, gone)
        for used in ("dumma::ld_matrix", "dumma::mma", "dumma::store_matrix",
                     "dumma::fill_fragment", "DFragment"):
            self.assertIn(used.replace("DFragment", "DUFragment"), gemm.replace(
                "DFragment", "DUFragment"), used)

    def test_opcodes_are_declared_once_in_config(self) -> None:
        # words[0] is a wire format shared by the runtime, the host encoder and
        # the op pack, so it has exactly one definition site.
        config = (OPS / "config.h").read_text()
        for name, value in (("kOpNoop", 0), ("kOpElementwise", 1), ("kOpGemm", 2),
                            ("kOpEmbedding", 3), ("kOpAttention", 5)):
            self.assertIn(f"{name} = {value}", config, name)
        self.assertIn("kOpcodeUnused = 4", config)
        self.assertIn("kMaxOpcode = 5", config)
        self.assertIn("constexpr bool opcode_known(int opcode)", config)

        # Every operator reads its opcode instead of repeating a literal.
        for header in _headers(OPS):
            text = header.read_text()
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith("static constexpr int opcode"):
                    self.assertIn("config::kOp", stripped, header.name)
                    self.assertNotRegex(stripped, r"=\s*\d", header.name)

    def test_the_host_encoder_agrees_with_the_cpp_table(self) -> None:
        # python/kernels/mk_stream.py builds the same instruction words; a drift
        # there would dispatch the wrong operator at runtime.
        config = (OPS / "config.h").read_text()
        stream = (ROOT / "python" / "kernels" / "mk_stream.py").read_text()
        for name, value, host in (
            ("kOpNoop", 0, "OPCODE_NOOP"),
            ("kOpElementwise", 1, "OPCODE_ELEMENTWISE"),
            ("kOpGemm", 2, "OPCODE_GEMM"),
            ("kOpEmbedding", 3, "OPCODE_EMBEDDING"),
            ("kOpAttention", 5, "OPCODE_ATTENTION"),
        ):
            self.assertIn(f"{name} = {value}", config, name)
            self.assertIn(f"{host} = {value}", stream, host)

    def test_no_operator_carries_a_mode_switch_any_more(self) -> None:
        for header in _headers(OPS):
            text = header.read_text()
            self.assertNotIn("OPS_MODE", text, header.name)

    def test_the_conv_kernel_is_a_plain_launch(self) -> None:
        kernel = (ROOT / "csrc" / "so2_conv.cpp").read_text()
        self.assertIn("__global__ void so2_conv_kernel", kernel)
        self.assertIn("<<<blocks_for(edges), kBlockThreads", kernel)
        self.assertIn('#include "so2_conv_math.h"', kernel)
        # No pages, no semaphores, no BaseOp: it is not a persistent op.
        for forbidden in ("State&", "semaphores()", "BaseOp"):
            self.assertNotIn(forbidden, kernel)

    def test_the_conv_arithmetic_exists_once(self) -> None:
        kernel = (ROOT / "csrc" / "so2_conv.cpp").read_text()
        math = (ROOT / "csrc" / "so2_conv_math.h").read_text()
        self.assertIn("sigmoid_content", math)
        self.assertNotIn("__expf", kernel)  # the gate expression lives in the math


