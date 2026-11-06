# DCU DPA4 GEMM

The current production operator is a contiguous, 2D FP32 `A[M,K] @ B[K,N]` implemented with DTK DUMMA. The native path is DCU-only and has no CPU, rocBLAS, or scalar-kernel fallback. A separate PyTorch oracle lives in `python/reference/` for explicit tests and comparisons. The next model milestone is single-card DPA4/SeZM inference returning energy and forces: implement required TileLang operators individually first, keep unported stages on ordinary PyTorch/kernels, and defer all fusion until the standalone operators and inference path are validated.

## Scope

- `python/kernels/`: production GEMM API and DCU extension checks.
- `python/kernels/tilelang_ops/`: optional TileLang operator-development package; currently includes a standalone Bessel radial-basis op with a PyTorch autograd wrapper.
- `python/reference/`: explicit PyTorch GEMM oracle; not imported as a production fallback.
- `csrc/gemm/gemm.cpp`: DUMMA FP32 16x16x4 kernel with masked M/N/K tails.
- `include/`: `.h` headers for the four DUMMA API wrappers, manual-based MMA shape selectors, and shared tile/LDS/cooperative-load/synchronization helpers.
- `tests/`: Python contracts, host-side helper contracts, and opt-in DTK/DCU accuracy tests.
- `doc/`: GEMM kernel and tile/LDS execution notes.

See [DMK current kernels](doc/dpa4-kernels.md) for implemented kernels and TileLang inference-development status, and [DeepMD-kit DPA4 operators](doc/megakernel.md) for framework dataflow, operator definitions, fused CUDA-op precedents, and distributed halo exchange.

## Python API

```python
import torch
from kernels import gemm
from reference import gemm as reference_gemm

a = torch.randn(19, 7, device="cuda", dtype=torch.float32)
b = torch.randn(7, 23, device="cuda", dtype=torch.float32)
y = gemm(a, b)                 # requires a supported DCU and built DUMMA extension
expected = reference_gemm(a, b)  # explicit PyTorch oracle
```

The first operator contract requires two contiguous 2D FP32 tensors on the same supported DCU, with matching inner dimensions. It is forward-only and does not support batching, transposes, bias, alpha/beta/C, or autograd. The oracle is available through `PYTHONPATH=python`; `kernels.gemm` never calls it implicitly.

## TileLang operator development

The first milestone is single-card DPA4/SeZM inference with **standalone, non-fused TileLang operators**. Only operators needed by the selected inference configuration will be implemented one by one; other stages remain on ordinary PyTorch/kernels. The workspace is `python/kernels/tilelang_ops/`: each operator lives in its own module with separate forward/backward kernels and a `torch.autograd.Function` entry point. The first op is raw Bessel radial basis; its backward propagates force gradients to distances, while the C3 envelope remains a separate PyTorch operation. In DeepMD-kit `pt_expt`, opt in with `DP_TILELANG_RADIAL_BASIS=1`; unported operators still use PyTorch. The package is not re-exported by `kernels` and does not add TileLang as a required dependency. Fusion is deferred until individual operator correctness and the end-to-end inference path are validated. Multi-rank ghost/halo exchange is outside the initial single-card scope.

Run the Python contract and reference tests:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests/python -v
```

## Build and test on DTK/DCU

The extension requires HIP-enabled PyTorch, the DTK HIP compiler, and the external `<du_mma.h>` header from a compatible DTK installation. The FP32 K=4 DUMMA shape is enabled for gfx926, gfx928, gfx936, and gfx938. `gemm.cpp` contains HIP syntax despite the `.cpp` suffix; setup compiles native `.cpp` files with `hipcc` (override with `DPA4_HIP_CXX`). The project does not vendor DUMMA.

```bash
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
DPA4_BUILD_HIP=1 DPA4_ENABLE_DUMMA=1 DPA4_HIP_CXX=hipcc MAX_JOBS=4 \
  python setup.py build_ext --inplace
python -m unittest discover -s tests/csrc -v
```

For the direct DUMMA instruction test, set `DPA4_RUN_INSTRUCTION_TESTS=1` and `DPA4_TEST_DUMMA=1` when running the C++ test suite. Native tests require a supported DCU; a non-DCU environment can run the Python oracle and host header tests but cannot verify the kernel.
