"""Host header contracts; opt-in real HIP/DUMMA compile and execution."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class InstructionHeaderTests(unittest.TestCase):
    def test_dumma_build_flag_requires_hip(self):
        env = dict(os.environ, DPA4_BUILD_HIP="0", DPA4_ENABLE_DUMMA="1",
                   PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, str(ROOT / "setup.py"), "--name"],
                                cwd=ROOT, env=env, capture_output=True, text=True,
                                timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DPA4_ENABLE_DUMMA=1 requires DPA4_BUILD_HIP=1", result.stderr)

    def test_direct_dumma_header_rejects_host_enable(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        self.assertIsNotNone(compiler)
        result = subprocess.run([
            compiler, "-std=c++17", "-DDPA4_ENABLE_DUMMA=1", "-x", "c++",
            "-fsyntax-only", "-I", str(ROOT / "include"), "-",
        ], input='#include "ops/group/dumma.cuh"\n',
            capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DPA4_ENABLE_DUMMA requires the DTK HIP compiler", result.stderr)

    def test_headers_are_self_contained(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        self.assertIsNotNone(compiler)
        headers = sorted((ROOT / "include").rglob("*.cuh"))
        self.assertGreaterEqual(len(headers), 10)
        for header in headers:
            relative = header.relative_to(ROOT / "include").as_posix()
            with self.subTest(header=relative):
                result = subprocess.run([
                    compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-x", "c++", "-fsyntax-only", "-I", str(ROOT / "include"), "-",
                ], input='#include "' + relative + '"\nint main() {}\n',
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_hardware_mma_does_not_fallback_on_host(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        self.assertIsNotNone(compiler)
        result = subprocess.run([
            compiler, "-std=c++17", "-DDPA4_ENABLE_DUMMA=1", "-x", "c++",
            "-fsyntax-only", "-I", str(ROOT / "include"), "-",
        ], input='#include "ops/ops.cuh"\n', capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("DPA4_ENABLE_DUMMA requires the DTK HIP compiler", result.stderr)

    def test_host_contracts(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        self.assertIsNotNone(compiler, "C++17 compiler is required for header tests")
        with tempfile.TemporaryDirectory(prefix="dpa4-instructions-") as tmp:
            binary = str(Path(tmp) / "host-test")
            result = subprocess.run([
                compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(ROOT / "include"),
                str(ROOT / "tests/csrc/instructions_host.cpp"), "-o", binary,
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("host instruction contracts passed", result.stdout)

    def test_invalid_reference_shape_is_rejected(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        self.assertIsNotNone(compiler)
        source = '''#include "ops/ops.cuh"
int main() {
    dcu::register_tile<float, 2, 3> a;
    dcu::register_tile<float, 4, 2> b;
    dcu::register_tile<float, 2, 2> c;
    dcu::thread::mma_reference(c, a, b, c);
}
'''
        result = subprocess.run([
            compiler, "-std=c++17", "-x", "c++", "-fsyntax-only",
            "-I", str(ROOT / "include"), "-",
        ], input=source, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mma_reference", result.stderr)
        self.assertIn("no matching function", result.stderr)


@unittest.skipUnless(os.environ.get("DPA4_RUN_INSTRUCTION_TESTS") == "1",
                     "set DPA4_RUN_INSTRUCTION_TESTS=1 on DCU to compile/run HIP")
class HIPInstructionTests(unittest.TestCase):
    def test_real_device(self):
        compiler = shutil.which(os.environ.get("HIPCXX", "hipcc"))
        self.assertIsNotNone(compiler, "requested HIP test but hipcc is unavailable")
        with tempfile.TemporaryDirectory(prefix="dpa4-instructions-hip-") as tmp:
            binary = str(Path(tmp) / "hip-test")
            flags = shlex.split(os.environ.get("DPA4_INSTRUCTION_HIP_FLAGS", ""))
            if os.environ.get("DPA4_TEST_DUMMA") == "1":
                flags.append("-DDPA4_ENABLE_DUMMA=1")
            result = subprocess.run([
                compiler, "-std=c++17", "-O2", *flags,
                "-I", str(ROOT / "include"),
                str(ROOT / "tests/csrc/instructions_hip.cpp"), "-o", binary,
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("HIP instruction contracts passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
