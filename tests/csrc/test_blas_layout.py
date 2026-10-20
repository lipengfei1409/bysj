import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class BLASLayoutTests(unittest.TestCase):
    def test_blas_build_flag_requires_hip(self):
        env = dict(os.environ, DPA4_BUILD_HIP='0', DPA4_BUILD_ROCBLAS='1',
                   PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, str(ROOT / 'setup.py'), '--name'],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('DPA4_BUILD_ROCBLAS=1 requires DPA4_BUILD_HIP=1', result.stderr)

    def test_headers_without_dtk(self):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        self.assertIsNotNone(compiler)
        result = subprocess.run([
            compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-x', 'c++',
            '-fsyntax-only', '-I', str(ROOT / 'include'), '-',
        ], input='#include "blas/rocblas.hpp"\nint main() {}\n',
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_row_major_layout_mapping(self):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory(prefix='dpa4-blas-layout-') as tmp:
            binary = str(Path(tmp) / 'layout-test')
            result = subprocess.run([
                compiler, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                '-I', str(ROOT / 'include'),
                str(ROOT / 'tests/csrc/blas_layout.cpp'), '-o', binary,
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([binary], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
