# SO(2) 卷积：`csrc/` 下的特定优化 kernel

## 目录：`ops/` 是工厂，`dmk/` 是持久化运行时

| 目录 | 放什么 | 命名空间 | 依赖方向 |
| --- | --- | --- | --- |
| `csrc/ops/` | **算子模板、工厂与共享内存契约**：`op.h`（`BaseOp` + `make_op`）、`state.h`（`Page` / `State` / 同步原语）、`config.h`（页/波/LDS 契约）、`lds.h`、`copy.h`、`tensor.h`、`tile.h`、`gemm.h`、`dumma.h`、以及通用算子 `attention.h`、`embedding.h`、`noop.h`、`elementwise.h` | **`ops::`**（`ops::BaseOp`、`ops::State`、`ops::config`、`ops::lds`、`ops::tile`…） | **不引用 `dmk/` 下任何东西**（连 `#include` 都不许） |
| `csrc/dmk/` | **持久化调度**：`megakernel.h/.cpp`（VM 与 launch）、`state.h`（`Role` / `worker_loop` / 页分配置换）、`controller.h` | `dmk::mk`（用 `namespace config = ops::config;`、`using ops::State;` 这类别名引用 ops） | 可以 include `ops/`（它是驱动方） |

**`State` 为什么在 `ops/`**：它是运行时与算子之间的**接口**（页、指令槽、信号量原语），算子必须
照着它写，所以它不能住在算子被禁止引用的目录里。调度那一半（角色选择、worker 循环、页分配置换）
留在 `dmk/state.h`，`#include "ops/state.h"`。这条单向依赖由 `tests/python/test_ops_dmk_layout.py`
钉住：`ops/` 下出现任何 `#include "dmk/` 或 `namespace dmk::` 都会失败。

这条方向由 `tests/python/test_ops_dmk_layout.py` 钉住（谁都不许裸 `#include "op.h"`，跨目录必须写全路径）。

## 分工：`ops/` 只放通用算子，`csrc/*.cpp` 放特定优化

| 目录 | 放什么 | 怎么跑 |
| --- | --- | --- |
| `csrc/ops/` | **通用算子**，全部按持久化调度编写：`noop`、`elementwise`、`embedding`、`gemm`、`attention`，加契约头（`op.h`、`state.h`、`config.h`、`lds.h`、`copy.h`、`tensor.h`、`tile.h`、`dtype.h`、`dumma.h`） | 进 op pack，由 `dmk/` 的 VM 驱动 |
| `csrc/*.cpp` | **模型特定优化**：`so2_conv.cpp`（本文件）、`gie.cpp`（zonal scatter / GIE）等 | 普通 kernel launch，不做成持久化调度 |
| `csrc/dmk/` | 持久化调度本身：`megakernel`、`state`、`controller` | 驱动 `ops/` 里的算子 |

理由是收益来源不同：通用算子受益于 megakernel 的指令流水与页机制；而 conv / GIE / zonal scatter
这类算子的收益来自**把整段 span 塞进一个 kernel**（去掉中间张量），它的并行结构（一个 wave 一条边、
lane = channel）与 VM 的"一条 instruction 一个算子调用"并不契合——把它做成持久化算子只会多一层
页搬运，所以在 `csrc/` 里单独写。

## 实现：一个普通 kernel

`csrc/so2_conv.cpp` 240 行，`csrc/so2_conv_math.h` 108 行（算术）。线程映射：**一个 wave64 一条边，
lane = channel**，所以每次 HBM 访问是一条 64-float 的合并行；一个 block 带 4 条边并行，跨步覆盖
任意边数。中间量 `xl / z / gate` 全在寄存器。

```bash
# 构建（和顶层其它算子一起）
MAX_JOBS=4 python3 setup.py build_ext --inplace
# 数值（DCU）
PYTHONPATH=python python3 -m unittest discover -s tests/csrc/accuracy -p 'test_so2_conv*' -v
```

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

## 故意不在这里的部分

| 缺什么 | 归谁 | 为什么 |
| --- | --- | --- |
| **目标节点归约**（边消息 → 原子求和，无原子 CSR） | `kernels.segment_sum` | 这是逐边算子与归约算子的组合方式，不是这个 span 的一部分 |
| **反向** | 尚未写 | 先测前向；反向正是 ridge 咬得最狠的地方（[`so2-conv-hip.md`](so2-conv-hip.md) §6.4） |
| **Wigner run 打包** | 现有 `dpa4_wigner_runs` 产出稠密 `R` | 让这个 kernel 吃稠密矩阵，不在里面重推块对角代数 |

## 测试

| 文件 | 在哪跑 | 钉住什么 |
| --- | --- | --- |
| [`tests/python/test_ops_dmk_layout.py`](../tests/python/test_ops_dmk_layout.py) | 本机 | **分工**：`ops/` 只含通用算子（15 个头文件，固定清单）、其中没有 `OPS_MODE` 残留；`so2_conv.cpp` 是普通 launch（`__global__` + `<<<>>>`），不含 `State&`/`semaphores()`/`BaseOp`；算术只在 `so2_conv_math.h` 出现一次 |
| [`tests/python/test_so2_conv_reference.py`](../tests/python/test_so2_conv_reference.py) | 本机 | **数学**：`kernel_order`（逐边逐通道的直译循环）与 `reference`（批量 einsum）互相吻合；正交旋转 + 零核 → 恒等；零 content 核 → 残差项消失；**content/gate 互换必须改变结果**（钉住两个核的顺序）；包络权重在回旋转之后乘；头文件里的阶段顺序与 oracle 一致 |
| [`tests/csrc/accuracy/test_so2_conv_accuracy.py`](../tests/csrc/accuracy/test_so2_conv_accuracy.py) | DCU（待接入） | 与 oracle 对拍：生产形状 `(16,10,4,C=64)`、窄通道 `C=5`（空闲 lane 不许写）、边数 `1/3/5/7`（覆盖 `kConsumerWaves=4` 的每种尾部相位）、单层小形状、确定性 |

两个 oracle 不是冗余：写测试时正是**批量 einsum 把 `err,erc->erc` 写成了取对角**（torch 对重复下标按对角处理），
逐边循环立刻以 0.67 的误差报了出来。这类错误在单个 oracle 下会静默通过。

## 验证路径（与 hipified 版共用同一套门）

```bash
# 1. 结构与数学（本机即可）
PYTHONPATH=python python3 -m unittest tests.python.test_ops_dmk_layout tests.python.test_so2_conv_reference -v

# 2. 数值：接进 megakernel 后，与稠密参考对拍（DCU）
#    参考：python/reference/*（dense 路径），容差 rtol=2e-5 / atol=2e-6
#    以及现有 so2_conv HIP 端口的上游测试（10 项 / 55 subtest）作为交叉验证

# 3. 性能：同一段边的设备时间 vs hipified 版（864 ms）与稠密（~690 ms）
```

**当前状态：代码已写、契约测试通过（10 项），但未编译、未接进 megakernel、无数值验证。**
下一轮的第一步应当是在 DCU 上把它编过（`hipcc -c` 单文件），第二步才是接 op pack。
