# SO(2) 卷积：megakernel 算子 + 普通算子两个入口

## 目录：`ops/` 是工厂，`dmk/` 是持久化运行时

| 目录 | 放什么 | 命名空间 | 依赖方向 |
| --- | --- | --- | --- |
| `csrc/ops/` | **算子模板、工厂与共享内存契约**：`op.h`（`BaseOp` + `make_op`）、`state.h`（`Page` / `State` / 同步原语——钩子拿到的那个句柄）、`config.h`（页/波/LDS 契约）、`lds.h`、`copy.h`、`tensor.h`、`tile.h`、`gemm.h`、`dumma.h`、`attention.h`、`embedding.h`、`noop.h`、`elementwise.h`、`so2_conv_op.h`、`so2_conv_math.h` | **`ops::`**（`ops::BaseOp`、`ops::State`、`ops::config`、`ops::lds`、`ops::tile`、`ops::so2`…；SO(2) 卷积辅助函数收在 `ops::so2_conv`） | **不引用 `dmk/` 下任何东西**（连 `#include` 都不许） |
| `csrc/dmk/` | **持久化调度**：`megakernel.h/.cpp`（VM 与 launch）、`state.h`（`Role` / `worker_loop` / 页分配置换）、`controller.h` | `dmk::mk`（用 `namespace config = ops::config;`、`using ops::State;` 这类别名引用 ops） | 可以 include `ops/`（它是驱动方） |

**`State` 为什么在 `ops/`**：它是运行时与算子之间的**接口**（页、指令槽、信号量原语），算子必须
照着它写，所以它不能住在算子被禁止引用的目录里。调度那一半（角色选择、worker 循环、页分配置换）
留在 `dmk/state.h`，`#include "ops/state.h"`。这条单向依赖由 `tests/python/test_ops_dmk_layout.py`
钉住：`ops/` 下出现任何 `#include "dmk/` 或 `namespace dmk::` 都会失败。

这条方向由 `tests/python/test_ops_dmk_layout.py` 钉住（谁都不许裸 `#include "op.h"`，跨目录必须写全路径）。

## 两种模式：编译期开关 `OPS_MODE`

每个算子都要能以两种形态编译，形态由**编译期**参数选：

| 开关 | 形态 | 依赖 | 入口 |
| --- | --- | --- | --- |
| `OPS_MODE_PERSISTENT`（默认，`0`） | `BaseOp` 子类，由 `dmk/` 的 VM 驱动 | `ops/op.h`、`ops/state.h` | op pack + instruction |
| `OPS_MODE_ORDINARY`（`1`） | 普通 `__global__` kernel + 内联 launch 辅助 | 只有 `hip/hip_runtime.h` | `<<<>>>`，和其它 kernel 一样 |

用法（翻译单元级）：

```cpp
#define OPS_MODE OPS_MODE_ORDINARY
#include "ops/so2_conv_op.h"     // 只编普通形态
```

或整次构建：`-DOPS_MODE=OPS_MODE_ORDINARY`。默认是持久化形态，所以"没想过这件事"的
翻译单元拿到的是它几乎肯定属于的那一种。定义在 [`csrc/ops/mode.h`](../csrc/ops/mode.h)，
写错值会 `#error`。

为什么必须是编译期的：两种形态在运行期没有任何共同点——持久化形态要 `State&` 与页契约，
普通形态要 grid/block 几何。同一次构建里混用会让同一个类出现两份定义。

**当前状态**：`OPS_MODE` 机制已就位；**SO(2) 卷积的两种形态收在同一个类里**：

```cpp
struct So2Conv {                                  // 一个类，两种形态
    static constexpr int kEdgeWaves, kBlockThreads;
    static unsigned blocks(long edges);
#if OPS_MODE == OPS_MODE_ORDINARY
    static void launch(pack, edges, dim, red, layers, channels, stream);
#endif
#if OPS_MODE == OPS_MODE_PERSISTENT
    template <typename Globals> class Op final : public BaseOp<Globals> { ... };
#endif
};
```

`csrc/so2_conv.cpp` 只剩 torch 胶水（校验、分配、`So2Conv::launch(...)`），kernel 体与几何
参数都在类里。`__global__` 函数仍是命名空间级自由函数——"静态成员能否是 `__global__`"是编译器的
细节，不值得拿构建去赌；`launch()` 已经把入口收进类里。其余五个算子（elementwise / embedding /
gemm / attention / noop）目前只有持久化形态，转换时照同一个模式加 `#if` 分支即可。

`tests/python/test_ops_dmk_layout.py` 钉住四件事：开关是单一编译期定义（默认持久化、非法值
`#error`）、SO(2) 头文件里两种形态都在、普通形态**看不到** `ops/state.h` / `ops/op.h`、
`csrc/so2_conv.cpp` 里不再有 `__global__`。

## 同一份算术，两个入口（当前只接普通入口）

```
csrc/ops/so2_conv_math.h        ← 唯一的算术（rotate / gated layer / rotate back）
        ├── csrc/ops/so2_conv_op.h     持久化接入：BaseOp 子类，进 op pack，由 VM 驱动
        └── csrc/dmk/so2_conv.cpp  普通接入：一次 launch（**只在 dmk 子树里构建**）
```

**当前只接普通入口。** 也就是说：

| 位置 | 现状 |
| --- | --- |
| `csrc/so2_conv.cpp` | **在顶层 `csrc/`**，和 `rmsnorm.cpp`、`segment_sum.cpp` 一样编译 |
| 顶层 `setup.py` 的 `sources` / `include_dirs` | 含 `csrc/so2_conv.cpp`，`include_dirs=["csrc"]`（`ops/...` 的包含路径） |
| `csrc/ops.h` / `csrc/bindings.cpp` | `so2_conv_hip` 声明 + `m.def("so2_conv", ...)` |
| `python/kernels/so2_conv.py` | `kernels.so2_conv` 公开导出 |
| **持久化算子** `ops/so2_conv_op.h` | **只有源码**：没有 launch 路径、不进任何构建 |
| `csrc/dmk/setup.py`、`python/kernels/dmk/` | **不存在** |

```bash
# 构建（和顶层其它算子一起）
MAX_JOBS=4 python3 setup.py build_ext --inplace

# 数值（DCU 上跑，CPU 上 skip）
PYTHONPATH=python python3 -m unittest discover -s tests/csrc/accuracy -p 'test_so2_conv*' -v
```

现在这样就够用的理由：普通入口**不需要** VM 就能把 `ops/so2_conv_math.h` 的算术语义验证掉，
而这正是下一步最需要知道的事（64-lane 布局的数学对不对、100M 量级的边数下什么速度）。等持久化
路径真正接线时，它的 A/B 对照也是这个普通入口——两者共用一份算术，差别只在调度。

两条边界由 `tests/python/test_ops_dmk_layout.py` 钉住：普通入口**必须**在顶层构建里，持久化算子
**必须不在**（`So2ConvOp` 不许出现在 `setup.py` / `ops.h` / `bindings.cpp`，且不许有 dmk 侧构建）。

## 持久化算子（`ops/so2_conv_op.h`）

手写的 HIP 版本，按 `BaseOp` 模板写：一个 instruction 就是**一次算子调用**
（一段边 × 一层 SO(2) 卷积，前向），`run()` 自己拥有 `instruction` 内的 tile 循环。

## 为什么重写而不是继续改 hipified 版

上游 kernel 的并行单元是 **32 lane**（CUDA warp），而 DCU 一个 wave 是 **64 lane**。hipified 移植
为了对齐 32，保留了每 warp 26×33 floats 的私有 staging 和 32-lane 的位序约定
（见 [`so2-conv-hip.md`](so2-conv-hip.md) §3）。这个版本把**lane 轴直接取成 channel 轴**：

```cpp
const int channel = lane;   // C = 64 -> 一个 wave 正好一个张量行
```

于是：

| | hipified（32-lane） | 本版本（64-lane） |
| --- | --- | --- |
| 每波处理的量 | 一个 warp 内 32 lane 拼 tile | **64 个 channel，宽度正好 256 B** |
| HBM 访问 | 需要 packed-lane staging 补齐 | 天然合并：一次 64-float 行 |
| 每 warp 私有 LDS staging | 26×33 floats（13.7 KB/块） | **0**（表按 wave 存页） |
| 峰值驻留瓶颈 | smem 24.2 KB → 2 块 | 页机制（6×8 KB），不再靠 smem 压 tile |

## 计算（一条边、一个 channel，全在寄存器里）

```
x[d]        = 源节点特征 (DIM,)
xl[r]       = Σ_d R[e][r][d] · x[d]                   旋转到局部系
for layer in 0..L-1:
  z[r]      = Σ_r' K[e][layer][r][r'] · xl[r']         块对角 content
  gate[r]   = Σ_r' G[e][layer][r][r'] · xl[r']         块对角 gate
  xl[r]    += silu(z[r]) · sigmoid(gate[r])            门控 + 残差
y[d]        = Σ_r R[e]ᵀ[d][r] · xl[r]                 回旋转
msg[e][d]   = alpha[e] · y[d]                          包络/注意力权重
```

中间量 `xl / z / gate` **从不落盘**——稠密图在这段里要物化 5 个 `(E, RED, C)` 张量。

## instruction 字与页布局

| 字 | 含义 | 页 | 内容 |
| ---: | --- | --- | --- |
| 0 | opcode (2) | `lid w` | 第 w 个 consumer wave 的表：`R`(red·dim)、`Rᵀ`(dim·red)、每层 `K,G`(layers·2·red²)、`alpha` |
| 1 | edge_begin | 4,5 | 未用，`load()` 立即释放 |
| 2 | edge_end | | |
| 3 | dim | | |
| 4 | red | | |
| 5 | layers | | |
| 6 | channels | | |

页预算：`2·red·dim + 2·layers·red² + 1` floats。生产形状 `(16, 10, 4)` → **1121 floats = 4484 B**，
8082 B 的页还剩一半（`static_assert` 钉住；`lmax=4` 的 `(25,13,4)` 也过）。

信号量：`[0] tables_ready`（loader→run，count 1）、`[1] msg_done`（每个 consumer wave 各 1 次，run→store）。

## 故意不在这里的部分

| 缺什么 | 归谁 | 为什么 |
| --- | --- | --- |
| **目标节点归约**（边消息 → 原子求和，无原子 CSR） | `SegmentSumOp`，另一条 instruction | 这是 megakernel 的拆分方式，不是融合 kernel 的单 span 结构 |
| **反向** | 尚未写 | 先测前向；反向正是 ridge 咬得最狠的地方（[`so2-conv-hip.md`](so2-conv-hip.md) §6.4） |
| **Wigner run 打包** | 现有 `dpa4_wigner_runs` 产出稠密 `R` | 让这个 op 吃稠密矩阵，不在 kernel 里重推块对角代数 |

## 接入需要三步（都在另一侧，本文件不改别人的代码）

1. `Globals` 加一个成员：`So2ConvTensorPack so2;`；
2. kernel 的 op pack 里加 `mk::So2ConvOp<Globals>`（`dmk/megakernel.cpp` 的 `mk_internal<...>` 列表）；
3. 发射端用 `mk::encode_so2_conv_instruction(words, begin, end, dim, red, layers, channels)`
   填 instruction，并先调用 `mk::so2_conv_tables_fit(dim, red, layers)` 校验页预算。

## 测试

| 文件 | 在哪跑 | 钉住什么 |
| --- | --- | --- |
| [`tests/python/test_so2_conv_op.py`](../tests/python/test_so2_conv_op.py) | 本机 | **模板合规**：五个钩子齐备、无析构（必须 trivially destructible）、构造函数吃指令字、opcode 在 dmk 表里唯一（0/1/2/3/4/5）；页预算 `(16,10,4)=1121 floats`、指令字 0–6 与编码器一致、信号量计数、`channel = lane` 的 lane 映射、缺失项已声明 |
| [`tests/python/test_so2_conv_reference.py`](../tests/python/test_so2_conv_reference.py) | 本机 | **数学**：`kernel_order`（逐边逐通道的直译循环）与 `reference`（批量 einsum）互相吻合；正交旋转 + 零核 → 恒等；零 content 核 → 残差项消失；**content/gate 互换必须改变结果**（钉住两个核的顺序）；包络权重在回旋转之后乘；头文件里的阶段顺序与 oracle 一致 |
| [`tests/csrc/accuracy/test_so2_conv_accuracy.py`](../tests/csrc/accuracy/test_so2_conv_accuracy.py) | DCU（待接入） | 与 oracle 对拍：生产形状 `(16,10,4,C=64)`、窄通道 `C=5`（空闲 lane 不许写）、边数 `1/3/5/7`（覆盖 `kConsumerWaves=4` 的每种尾部相位）、单层小形状、确定性 |

两个 oracle 不是冗余：写测试时正是**批量 einsum 把 `err,erc->erc` 写成了取对角**（torch 对重复下标按对角处理），
逐边循环立刻以 0.67 的误差报了出来。这类错误在单个 oracle 下会静默通过。

## 验证路径（与 hipified 版共用同一套门）

```bash
# 1. 结构/契约（本机即可）
PYTHONPATH=python python3 -m unittest tests.python.test_so2_conv_op -v

# 2. 数值：接进 megakernel 后，与稠密参考对拍（DCU）
#    参考：python/reference/*（dense 路径），容差 rtol=2e-5 / atol=2e-6
#    以及现有 so2_conv HIP 端口的上游测试（10 项 / 55 subtest）作为交叉验证

# 3. 性能：同一段边的设备时间 vs hipified 版（864 ms）与稠密（~690 ms）
```

**当前状态：代码已写、契约测试通过（10 项），但未编译、未接进 megakernel、无数值验证。**
下一轮的第一步应当是在 DCU 上把它编过（`hipcc -c` 单文件），第二步才是接 op pack。
