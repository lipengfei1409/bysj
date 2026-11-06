# DMK 当前已实现的 Kernel

## 范围与状态

当前仓库从算子层重新开始，只保留一个原生算子：DUMMA FP32 GEMM。DeepMD-kit 中 DPA4/SeZM 的完整算子分解与定义见 [DeepMD-kit DPA4 算子](megakernel.md)；那份文档是框架数据流说明，不表示这些算子已在本仓库实现。

### TileLang 单卡推理开发状态

当前阶段先为单卡 DPA4/SeZM inference（能量与力输出）逐个实现独立 TileLang operator，不做任何通信/计算融合。`python/kernels/tilelang_ops/radial_basis.py` 是首个算子模块：实现 raw Bessel radial basis 及其 distance backward，并通过 `torch.autograd.Function` 暴露；C3 envelope 仍单独由 PyTorch 计算。DeepMD-kit `pt_expt` 路径用 `DP_TILELANG_RADIAL_BASIS=1` 显式启用。其余算子继续走普通 PyTorch/kernel；TileLang 不改变生产 GEMM API。首阶段不要求 stress/virial；多 rank ghost/halo exchange 也不在范围内。

生产 API 是 `kernels.gemm(a, b)`，定义为：

```text
C[M,N] = A[M,K] @ B[K,N]
```

输入必须是同一设备上的连续二维 FP32 DCU Tensor，且 `A.shape[1] == B.shape[0]`。当前接口是前向-only，不支持 batch、转置、bias、alpha/beta/C、autograd，也不提供 CPU、rocBLAS 或标量 kernel 的 fallback。独立 PyTorch oracle 位于 `python/reference/gemm.py`，测试时显式调用 `reference.gemm`。

## Native 调用链

```text
kernels.gemm
  -> kernels.native.require_dcu / kernels._C
  -> csrc/bindings.cpp: gemm
  -> csrc/gemm/gemm.cpp: gemm_dumma_hip
  -> gemm_dumma_kernel<dmk::group::mma_f32_16x16x4>
```

`gemm_dumma_hip` 检查设备、dtype、连续性、矩阵维度与 K 维匹配，然后按 M/N 输出 tiles 发射 HIP kernel。它只在 DUMMA build 开启时存在；找不到 extension、HIP runtime 或支持的 gfx 架构时，Python 接口报错，不转入 reference。

## Kernel 定义与执行

- **指令形状**：一个 64-lane wave/CTA 计算一个 `16x16` 的 FP32 输出 tile，每个 K step 为 4。
- **输入**：行主序 `A[16,4]`、`B[4,16]`；DUMMA fragment 的 lane 分布是不透明的，由 DTK DUMMA API 管理。
- **累加**：整个 wave 协同将 A/B tile 从 global memory 搬到 LDS，再调用 `du_load_matrix_sync` 和 `du_mma_sync`；累加器以 FP32 初始化为 0。
- **尾块**：cooperative load 将越界的 M/N/K 元素补零；DUMMA 本身不做 mask。输出先写入 LDS，再由线程按 M/N 边界做 masked global stores。
- **同步**：每个 K step 在 LDS load 后同步；DUMMA fragment load 完成后同步，防止下一步覆盖 LDS；DUMMA store 到 output LDS 后再次同步。当前路径是同步实现，不假设 async copy 或异步 MMA。
- **K=0**：不执行 K-loop，零初始化的累加器写出零矩阵；M 或 N 为零时不发射 kernel。

Native kernel 使用 `include/ops/group/dumma.h` 的 DTK facade、`include/ops/group/mma.h` 的 FP32 `16x16x4` shape、`include/ops/group/memory.h` 的 cooperative load、`include/ops/group/sync.h` 的 block barrier，以及 `include/common/tile.h` 的 row-major `matrix_view/shared_tile`。这些 header 是 GEMM 所需的共享原语，不是额外 DPA4 算子。

## DUMMA 函数与 MMA shape 选择

`include/ops/group/dumma.h` 以模板 facade 暴露 DTK 26.04 手册 §2 的四个函数名；fragment 类型仍由外部 `<du_mma.h>` 提供：

| Include API | 说明 |
|---|---|
| `dmk::dumma::du_fill_fragment(fragment&, value)` | 填充值到 fragment。 |
| `dmk::dumma::du_load_matrix_sync(fragment&, ptr, ldm)` | 加载 A/B；leading dimension 单位为元素，布局由 fragment 确定。 |
| `dmk::dumma::du_load_matrix_sync(fragment&, ptr, ldm, layout)` | 带显式布局加载累加器 C。 |
| `dmk::dumma::du_mma_sync(d, a, b, c)` | 执行 `D = A * B + C`。 |
| `dmk::dumma::du_store_matrix_sync(ptr, fragment, ldm, layout)` | 将结果写到 device-accessible memory。 |

`du_load_matrix_sync` 的两个 overload 对应手册中的一个 API。四个 facade 均直接调用 DTK 的同名 DUMMA 接口；必须由完整且收敛的 wave64 以一致参数调用。leading dimension 按元素计数。

`include/ops/group/mma.h` 提供 compile-time `select_mma_shape_t<A, B, Acc, K>`，尺寸统一为 `M=N=16`。以下 shape/type/架构组合来自手册 §3 表 3.1：

| A / B | Accumulator | M×N×K | 支持架构 |
|---|---|---:|---|
| FP32 / FP32 | FP32 | 16×16×4 | gfx926, gfx928, gfx936, gfx938 |
| FP32 / FP32 | FP32 | 16×16×8 | gfx928, gfx936, gfx938 |
| TF32 / TF32 | FP32 | 16×16×8 | gfx928, gfx936, gfx938 |
| FP16 / FP16 | FP32 | 16×16×16 | gfx928, gfx936, gfx938 |
| BF16 / BF16 | FP32 | 16×16×16 | gfx928, gfx936, gfx938 |
| signed INT8 / signed INT8 | INT32 | 16×16×32 | gfx928, gfx936, gfx938 |
| unsigned INT8 / unsigned INT8 | INT32 | 16×16×32 | gfx928, gfx936, gfx938 |
| packed S4 / packed S4 | INT32 | 16×16×64 | gfx936, gfx938 |
| packed U4 / packed U4 | INT32 | 16×16×64 | gfx936, gfx938 |
| FP64 / FP64 | FP64 | 16×16×4 | gfx926, gfx936, gfx938 |
| FP8 E4M3 / FP8 E4M3 | FP32 | 16×16×32 | gfx938 |
| FP8 E5M2 / FP8 E5M2 | FP32 | 16×16×32 | gfx938 |

例如，`select_mma_shape_t<mma_operand_type::fp32, mma_operand_type::fp32, mma_accumulator_type::fp32, 4>` 选择 FP32 `16x16x4` descriptor。shape descriptor 负责选择元数据，不代表这些 dtype 均已接入 DMK GEMM kernel；当前 kernel 只实例化 FP32 `16x16x4`。手册指出 TF32 输入需先转换/处理（可用 `__du_float_to_tf32`），S4/U4 需先将两个 4-bit 值打包为一个 char。

项目内所有 HIP header/source 使用 `.h` / `.cpp` 命名。`gemm.cpp` 仍是 HIP translation unit，构建时以 `hipcc` 编译，默认可通过 `DPA4_HIP_CXX` 指定编译器。

## 支持范围与构建

- 当前 DUMMA shape 是 true FP32（非 TF32）K=4，项目声明的支持目标为 `gfx926`、`gfx928`、`gfx936`、`gfx938`。
- `<du_mma.h>` 来自兼容 DTK 安装，不 vendoring 到本仓库；完整 wave64 必须一致参与每个 DUMMA 调用。
- 需要 HIP-enabled ROCm/DTK PyTorch、DTK HIP compiler 和 DUMMA header：

```bash
DPA4_BUILD_HIP=1 DPA4_ENABLE_DUMMA=1 python setup.py build_ext --inplace
```

- CPU 可运行 `python/reference` 与 Python 合约测试，也可验证 host header contract；这不能证明 DCU kernel 的正确性或性能。

关键代码：`python/kernels/gemm.py`、`python/kernels/native.py`、`csrc/bindings.cpp`、`csrc/gemm/gemm.cpp`、`include/ops/group/dumma.h`。测试：`tests/python/test_blas.py`、`tests/python/test_interfaces.py`、`tests/csrc/test_accuracy.py`、`tests/csrc/test_instructions.py`。
