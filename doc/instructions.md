# DCU 指令/fragment 层

这层是构建算子的 C++17 header-only 基础库，不是完整 GEMM，也不是新的 DPA4 Python API。所有 kernel 可包含 `#include "ops/ops.cuh"`，编译添加 `-I include`。extension 的 `setup.py` 已加入此搜索路径；无 Torch 依赖的独立 HIP kernel 也能使用。

## 分层与接口

| 文件 | 作用域/接口 | 数据合同 |
| --- | --- | --- |
| `common/config.cuh` | `compiled_arch`、`dumma_supported<Arch,T,K>` | 目标架构来自编译器，不来自 CPU；未知架构不启用 DUMMA |
| `common/tile.cuh` | `matrix_view<T>`、`register_tile<T,R,C>`、`shared_tile<T,R,C>` | row-major；view stride 单位为元素；register tile 是单线程完整 tile |
| `ops/thread/math.cuh` | `thread::fma(a,b,c)` | FP32/FP64，融合乘加 |
| `ops/thread/memory.cuh` | `fill`、`load_masked`、`store_masked`、`load`、`store` | 越界 load 为 0、store 不写；不做隐式 dtype 转换 |
| `ops/thread/mma.cuh` | `thread::mma_reference(d,a,b,c)` | `[M,K]×[K,N]+[M,N]→[M,N]`；标量 FMA 循环，**不是硬件 MMA** |
| `ops/group/wave.cuh` | `thread_rank/lane_id/wave_id`、`shuffle/shuffle_down`、`wave_sum/wave_max` | FP32；完整且收敛的硬件 wave；sum/max 广播给所有 lane |
| `ops/group/memory.cuh` | `group::load/store` | 全 block 合作式 global↔LDS 搬运；尾块 mask；不是异步 copy |
| `ops/group/sync.cuh` | `block_sync/block_fence/device_fence` | barrier 与内存 fence 分开；fence 不意味着其他线程已经完成 |
| `ops/group/dumma.cuh` | `dumma::fill_fragment/load_matrix_sync/mma_sync/store_matrix_sync` | 逐一转发手册的四类 API，保留原生 DUFragment 类型，不执行软件 fallback |
| `ops/group/mma.cuh` | `mma_f32_16x16x4/8`、`mma_f16_16x16x16` | 一个完整 wave64；FP32 累加；DTK 原生 opaque fragment |

`matrix_view` 不拥有内存。调用方应在 launch 前确认 `valid()`、底层分配容量、指针生命周期、地址空间和索引不溢出；`operator()` 本身不检查越界。空 view 可以 data=nullptr，但不可直接解引用。`shared_tile` 只是相同存储结构的别名，需要在 kernel 内明确声明 `__shared__`。`register_tile` 不保证一定驻留物理寄存器，过大的实例可能 spill。

## MMA：真实指令与参考运算不能混用

### 手册指令的一一对应封装

独立指令入口是 `ops/group/dumma.cuh`，不必依赖 `mma_16x16` 类。所有参数按照手册第 2 节传递：

| 本项目 API | DTK 原始 API | 作用 |
| --- | --- | --- |
| `dcu::dumma::fill_fragment(fragment, value)` | `du::dumma::du_fill_fragment` | 填充 fragment |
| `dcu::dumma::load_matrix_sync(fragment, ptr, ld)` | `du::dumma::du_load_matrix_sync` 三参数版本 | 加载 A/B，布局由 fragment 类型定义 |
| `dcu::dumma::load_matrix_sync(fragment, ptr, ld, layout)` | `du::dumma::du_load_matrix_sync` 四参数版本 | 加载累加矩阵 C |
| `dcu::dumma::mma_sync(d, a, b, c)` | `du::dumma::du_mma_sync` | collective D=A×B+C |
| `dcu::dumma::store_matrix_sync(ptr, fragment, ld, layout)` | `du::dumma::du_store_matrix_sync` | 写到 device-accessible 内存；layout 默认 row-major |

`DUFragment`、`matrix_a`、`matrix_b`、`accumulator`、`row_major` 直接使用原生类型的 using 声明，不另造 ABI。`row_major_memory` 是原生 `mem_row_major` 的别名值；其他原生类型/布局可通过 `dcu::dumma::native` 使用，不自行推测 fragment 内部布局。该薄层不限制在三个 convenience aliases 上，但也不会替其他类型做量化、打包、TF32 预处理或保证其硬件支持；使用者必须遵循手册表 3.1 和当前安装头文件的合同。

```cpp
#include "ops/group/dumma.cuh" // 编译加 -DDPA4_ENABLE_DUMMA=1

// 16x16x4 FP32，block=64、完整收敛 wave；输入与输出均为 device 指针。
__global__ void direct_dumma(const float* A, const float* B,
                             const float* C, float* D) {
    namespace dm = dcu::dumma;
    dm::DUFragment<dm::matrix_a, 16, 16, 4, float, dm::row_major> a;
    dm::DUFragment<dm::matrix_b, 16, 16, 4, float, dm::row_major> b;
    dm::DUFragment<dm::accumulator, 16, 16, 4, float> c, d;
    dm::load_matrix_sync(a, A, 4);
    dm::load_matrix_sync(b, B, 16);
    dm::load_matrix_sync(c, C, 16, dm::row_major_memory);
    // 若没有累加项，将上一行改为 dm::fill_fragment(c, 0.f)。
    dm::mma_sync(d, a, b, c);
    dm::store_matrix_sync(D, d, 16);
}
```

`ops/group/mma.cuh` 的固定 shape 便捷类已改为复用此薄层，并保留额外的架构/精度检查。直接使用原生 fragment 时，由 DTK 负责其支持判断，调用者负责确认当前 gfx；没有软件替代路径。

手册 2.4 的文字把 store 描述为 host 写回，但第 4 节示例传入的是 `d_batch`（device 内存），再单独 `hipMemcpyDeviceToHost`。本项目按该示例执行，store 不隐式搬回 CPU。

DTK 26.04《DUMMA 使用手册》表 3.1 和第 4 节示例提供 `du_mma.h`、`du::dumma::DUFragment`、`du_load_matrix_sync`、`du_fill_fragment`、`du_mma_sync`、`du_store_matrix_sync`。当前封装选择以下三个合同，不推断其他架构支持：

| 封装 | A/B | Accumulator | M,N,K | 文档支持的 gfx |
| --- | --- | --- | --- | --- |
| `mma_f32_16x16x4` | float | float | 16,16,4 | 926 / 928 / 936 / 938 |
| `mma_f32_16x16x8` | float | float | 16,16,8 | 928 / 936 / 938 |
| `mma_f16_16x16x16` | __half | float | 16,16,16 | 928 / 936 / 938 |

float 路径不使用 TF32 截断；FP16 路径由调用方显式转换 A/B，不能未经 E/F/V 精度验证替换现有 FP32 模型。暂不加入 BF16、FP8、整数和原始 MFMA builtin/inline assembly；这可以在确定真实 gfx 和 DTK 头文件/ISA 后独立扩展。

编译显式加 `-DDPA4_ENABLE_DUMMA=1`，否则硬件 MMA 类型不可用；缺 `du_mma.h`、目标架构或 shape 不支持时报编译错误，不自动改用 scalar loop。默认 aggregate header 不强制普通 ROCm 环境安装厂商 DUMMA。尚未在真实 DTK 上编译/执行这些封装。

### 使用示例

```cpp
#include "ops/ops.cuh"

// A[16,4], B[4,16], C/D[16,16], 全部 row-major。
// 一个 block=64 个线程，所有 lane 调用相同 collective、相同指针与 ld。
__global__ void tile_mma(const float* A, const float* B,
                         const float* C, float* D) {
    using op = dcu::group::mma_f32_16x16x4;
    op::a_fragment a;
    op::b_fragment b;
    op::accumulator c, d;
    op::load_a(a, A, 4);
    op::load_b(b, B, 16);
    op::load_c(c, C, 16); // 或 op::fill(c, 0.f)
    op::mma(d, a, b, c);
    op::store(D, d, 16); // D 是 device/global 或兼容的 LDS 指针，不是 host 回传
}
```

必须满足当前 DTK `du_mma.h` 的指针对齐、leading dimension 和地址空间约束；封装不检查它们。原生 fragment 内部元素排列不暴露给上层，不允许套用 ThunderKittens 的 warp32 fragment 索引。多个 wave 可以处理不同 tile，但每个 wave 内调用必须收敛、指针一致；边界条件通过 loader 补零，不能令部分 lane 跳过 MMA。`D` 与 `C` fragment 使用独立对象；只有 `thread::mma_reference` 明确支持 D/C 别名。

## 如何用于后续角色化 kernel

当前可执行的数据路径是：

```text
global A/B → group::load（尾块补零）→ LDS tile
          → block_sync → op::load_a/load_b → wave fragment
          → op::mma → accumulator → op::store → global D
```

- loader：组织 tile 地址、mask 与 LDS buffering。
- consumer：在完整 wave 内调用 DUMMA，不假设 native fragment 元素分布。
- storer：组织输出地址、尾块 staging、后处理及写回。
- controller/worker：`csrc/megakernel/runtime.cu` 中已有 opt-in edge-feature 队列执行器；selected runtime plan 可提交 `edge_features` 及有限依赖 closure，仍不支持完整 DPA4 task graph。

**当前 `group::load/store` 是全 block 合作接口，不是只给 loader/storer role 调用的接口。** 所有 block 线程必须执行 `block_sync()`；不能在 consumer 分支里独自执行全 block barrier。role specialization 需要另外实现带 participant 范围的 copy/barrier 和明确的 LDS buffer 所有权。也不要把 `device_fence()` 当作跨 block 的 grid barrier，或把它当作队列原子的 acquire/release 协议。

## 验证命令

在工程根目录，host 测试无需 Torch 或 HIP：

```bash
python3 -m unittest tests.csrc.test_instructions.InstructionHeaderTests -v
```

在已安装 DTK 的 DCU 上，先用 `rocminfo` 或 PyTorch `gcnArchName` 确认真实架构；下面的 gfx936 是示例，不能替所有机器固定此值。无需运行任何系统依赖覆盖命令。

```bash
source /opt/dtk/env.sh
# 不启用 DUMMA：先检查 HIP memory/FMA/wave 基元
DPA4_RUN_INSTRUCTION_TESTS=1 \
  python -m unittest tests.csrc.test_instructions -v

# DUMMA FP32 K=4（需真实 gfx 在支持列表中且 DTK 提供 du_mma.h）
DPA4_RUN_INSTRUCTION_TESTS=1 DPA4_TEST_DUMMA=1 \
  DPA4_INSTRUCTION_HIP_FLAGS='--offload-arch=gfx936' \
  python -m unittest tests.csrc.test_instructions -v

# 扩展测试 K=8 和 FP16 K=16，仅 gfx928/936/938；不能用于 gfx926
DPA4_RUN_INSTRUCTION_TESTS=1 DPA4_TEST_DUMMA=1 \
  DPA4_INSTRUCTION_HIP_FLAGS='--offload-arch=gfx936 -DDPA4_TEST_EXTENDED_DUMMA=1' \
  python -m unittest tests.csrc.test_instructions -v
```

设 `DPA4_RUN_INSTRUCTION_TESTS=1` 后，编译器缺失、编译失败、设备不可用或数值错误会使测试失败，不会 skip。未显式启用时实机测试 skip，不能据此宣称 DUMMA 验证通过。

独立运行/查看输出，含可选 HIP event microbenchmark：

```bash
mkdir -p build/instructions
hipcc -std=c++17 -O2 -I include \
  --offload-arch=gfx936 -DDPA4_ENABLE_DUMMA=1 \
  -DDPA4_TEST_EXTENDED_DUMMA=1 \
  tests/csrc/instructions_hip.cpp -o build/instructions/hip-test
DPA4_INSTRUCTION_BENCHMARK=1 build/instructions/hip-test
```

测试覆盖两个完整 wave 的 sum/max/shuffle、二维 block 索引、带偏移的合作搬运、边界不写、FP32/FP16 D=A×B+C、非零/零 C、不同 leading dimension 和 LDS 补零。FP16 oracle 使用已经量化的输入。微基准是在精度通过后测 **单 tile 完整 kernel（global/LDS/MMA/store）** 的 event 时间，含 launch 开销；不是纯 MMA 指令延迟，也不是 DPA4 加速率。生成 ISA 及矩阵指令是否实际发射仍须用该 DTK 的 dccobjdump/编译器产物检查。

## ThunderKittens 对照

参考本地 `ThunderKittens/include/types/register/rt.cuh` 的类型分层、`include/ops/group/mma/warp.cuh` 的 fragment/MMA 封装方式、`include/ops/group/memory/tile/global_to_shared.cuh` 的合作搬运方式、`include/ops/thread/util/sync.cuh` 的同步职责拆分。没有复制它的 NVIDIA PTX、TMA、WGMMA 或 mbarrier 实现。

当前不会自动改变 `csrc/gemm/linear.cu`：先在真实 DCU 验证这些基元，再实现 padded/tiled GEMM 和 tail epilogue，保持 Python 接口、FP32 基线与现有 backward 合同。
