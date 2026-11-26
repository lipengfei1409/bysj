# SO(2) 卷积的 megakernel 算子（`csrc/so2_conv_dmk/so2_conv_op.h`）

手写的 HIP 版本，按 `csrc/dmk/op.h` 的 `BaseOp` 模板写：一个 instruction 就是**一次算子调用**
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
2. kernel 的 op pack 里加 `mk::So2ConvOp<Globals>`；
3. 发射端用 `mk::encode_so2_conv_instruction(words, begin, end, dim, red, layers, channels)`
   填 instruction，并先调用 `mk::so2_conv_tables_fit(dim, red, layers)` 校验页预算。

## 测试

| 文件 | 在哪跑 | 钉住什么 |
| --- | --- | --- |
| [`tests/python/test_so2_conv_dmk_op.py`](../tests/python/test_so2_conv_dmk_op.py) | 本机 | **模板合规**：五个钩子齐备、无析构（必须 trivially destructible）、构造函数吃指令字、opcode 不撞 `demo_op`；页预算 `(16,10,4)=1121 floats`、指令字 0–6 与编码器一致、信号量计数、`channel = lane` 的 lane 映射、缺失项已声明 |
| [`tests/python/test_so2_conv_dmk_reference.py`](../tests/python/test_so2_conv_dmk_reference.py) | 本机 | **数学**：`kernel_order`（逐边逐通道的直译循环）与 `reference`（批量 einsum）互相吻合；正交旋转 + 零核 → 恒等；零 content 核 → 残差项消失；**content/gate 互换必须改变结果**（钉住两个核的顺序）；包络权重在回旋转之后乘；头文件里的阶段顺序与 oracle 一致 |
| [`tests/csrc/accuracy/test_so2_conv_dmk_accuracy.py`](../tests/csrc/accuracy/test_so2_conv_dmk_accuracy.py) | DCU（待接入） | 与 oracle 对拍：生产形状 `(16,10,4,C=64)`、窄通道 `C=5`（空闲 lane 不许写）、边数 `1/3/5/7`（覆盖 `kConsumerWaves=4` 的每种尾部相位）、单层小形状、确定性 |

两个 oracle 不是冗余：写测试时正是**批量 einsum 把 `err,erc->erc` 写成了取对角**（torch 对重复下标按对角处理），
逐边循环立刻以 0.67 的误差报了出来。这类错误在单个 oracle 下会静默通过。

## 验证路径（与 hipified 版共用同一套门）

```bash
# 1. 结构/契约（本机即可）
PYTHONPATH=python python3 -m unittest tests.python.test_so2_conv_dmk_op -v

# 2. 数值：接进 megakernel 后，与稠密参考对拍（DCU）
#    参考：python/reference/*（dense 路径），容差 rtol=2e-5 / atol=2e-6
#    以及现有 so2_conv HIP 端口的上游测试（10 项 / 55 subtest）作为交叉验证

# 3. 性能：同一段边的设备时间 vs hipified 版（864 ms）与稠密（~690 ms）
```

**当前状态：代码已写、契约测试通过（10 项），但未编译、未接进 megakernel、无数值验证。**
下一轮的第一步应当是在 DCU 上把它编过（`hipcc -c` 单文件），第二步才是接 op pack。
