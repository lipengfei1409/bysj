# 执行模型：persistent 调度还是 megakernel

结论先给：**两者不是二选一**。persistent 是"网格怎么摊到 CU 上"（固定 grid、block 内循环遍历
tile），megakernel 是"一个 kernel 里放多少个 op"（融合边界）。这张图上前者的红利很小，
后者的红利很大，但只在**一个段**里可行 —— `G2 槽位段`（580.0 ms / 47.6%）。

数据来自 `dpa4_inference_trace.json`，口径见 [`op-time-share.md`](op-time-share.md)。

## 1. 三条先把方向钉死的事实

| 事实 | 数字 | 推论 |
| --- | --- | --- |
| launch 不是瓶颈 | 设备忙碌比 **96.9%**（1218.6 / 1257.8 ms）；B2 省掉 150 次 launch，端到端只动 0.6 ms | "megakernel = 少 launch"在这里不成立 |
| 时间集中在大张量 | >46 MB 的 launch 只有 653 个（25%），占 **1104.5 ms（90.7%）**；≤E 的 848 个只占 12.2 ms（1.0%） | 把全部小算子合成一个 kernel 最多省 12 ms |
| 带宽已经用掉八成 | G2 段实测 **511 GB/s**（296.5 GB / 580 ms），630 GB/s 下限是 470.6 ms | 单靠"访存更快"只有 109 ms 空间，真正的钱在**减少 pass 数** |

所以 megakernel 的价值是**省下中间量往返 HBM 的字节**，不是省 launch，也不是提高带宽。

## 2. 按张量家族分池子，再决定执行模型

| 池子 | 内容 | ms | 占比 | 实测 GB/s | 建议执行模型 | 预计收益 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **G2** | 槽位段：`[E,3/4/9/10/15/16,64]` 的展开、残差、gate、以及 6 个 per-edge 小 `bmm`（`[E,10,10]`、`[E,10,16]`、`[E,16,10]`） | **580.0** | **47.6%** | 511 | **段级 megakernel**（fwd/bwd 分开写） | fwd −100…−125；bwd −270…−320 |
| G1 | 通道混合：`[E,384]` / `[E,640]` / `[E,256]` 的两个大 GEMM + `cat` + `copy_` | 225.6 | 18.5% | 409 | 不做 megakernel，只做 epilogue（属 DMK 线） | −20…−40 |
| G4 | 其他 E-major（`[E,10,1,64]` 的 `sum`、`[84,E]` 的 `mul`、`[E,16,1,1,64]` 的缩放与 gather） | 201.8 | 16.6% | 432 | 小轴归约换 lane 映射（C4）；缩放类并入 G2 | −25…−40 |
| G5 | node-major `[3000,16,·]` 的 gather/scatter | 113.5 | 9.3% | 587 | persistent + CSR 段归约（已在做） | −10…−20 |
| G9 | readout `[1,152,48]×[1,48,576000]` 等 | 85.5 | 7.0% | 296 | persistent 单算子 + `float4` | −15…−25 |
| G3 | 逐边标量（`[E]`/`[E,1]`/`[E,4]`），848 次 launch | 12.2 | 1.0% | 83 | 可选：合成 1 个 persistent kernel | −8…−10 |

## 3. 为什么只有 G2 值得 megakernel

**a. 驻留预算刚好放得下。** G2 的张量是 `[E, slot, 64]`，slot ≤ 16：
每条边 16 × 64 × 4 B = **4 KB**。而 wave64 的 lane 正好可以一个通道一个 lane，
于是"一条边 = 一个 wave"，16 个槽就是每条 lane 16 个 VGPR —— **连 LDS 都不用**；
per-edge 的 10×10 mixing 矩阵只有 400 B，走 LDS 广播即可。
反过来看 G1：`[E,384]` 是 1.5 KB/边、`[E,640]` 是 2.5 KB/边，几个叠起来就超出寄存器/LDS。

**b. 边界干净，没有跨 block 依赖。** G2 的 383 次 launch 全部作用在同一批边流张量上，
段内不需要全局同步；G1 的 `cat([E,256] ⊕ [E,384])` 则需要等上一步全部算完。

**c. 段内 pass 数可以砍一个量级。** 现在这条链上每个 op 都要把 `[E,10,64]`（465 MB）
读一遍写一遍，一段里十几个 op 就是十几 GB；把"mixing → gate → 残差"留在寄存器里，
同样的语义只要 1 读 1 写。这是 −270 ms 那类数字的唯一来源。

**d. 反向必须单独写。** G2 反向 416.2 ms / 218.2 GB，结构是转置 bmm + gate scatter
（`index_add_ [1,E,3,64] ⬅ [1,E,9,64]`），和前向不是同一套算子，不能指望一个 kernel 两用。

## 4. 不做全模型 megakernel 的理由

1. 驻留量翻倍：G1 的 `[E,384]`/`[E,640]` 必须与 G2 同时在飞，tile 只能降到 6–8 边，wave 内并行度不够。
2. 需要全局 barrier：`cat` 是跨 block 的依赖，纯 persistent + 全局同步的复杂度远高于收益。
3. GEMM 已经不是瓶颈项：两个大 `bmm` 每次 3.5 ms、按 2MNK 折算约 15 TFLOP/s，属于分块/epilogue 问题，不是调度问题。

## 4.5 上游其实已经实现了这个 megakernel（重要）

DeepMD 自带一整套融合算子（5 套实现层、约 20 个融合点），逐项清单见
[`deepmd-fusions.md`](deepmd-fusions.md)。这里只留与本决策相关的两个事实：

native 那几套只在 `if(USE_CUDA_TOOLKIT AND DEEPMD_TORCH_HAS_CUDA)` 下进入构建
（`source/op/pt/CMakeLists.txt`），所以 DCU 构建里根本没有
`torch.ops.deepmd.dpa4_so2_conv`，各 `op_available()` 恒为 `False`，dispatch 静默回退稠密路径 ——
这就是我们 profile 到的 1218.6 ms。

`deepmd/pt_expt/kernels/cuda/dpa4/so2_conv.py` 绑定的是 `deepmd::dpa4_so2_conv`
（`source/op/pt/dpa4/so2_conv_fwd_c{32,64}_l{1..6}.cu` 与对应的 `_bwd_`），它一次性完成
**一个 `SO2Convolution` 的整条 per-edge 路径**：attention logits + 门控分段 softmax +
Wigner 旋转 + radial 混合 + gated mixing stack + 逆旋转 + attention 加权 + 目标归约 +
head gate，且 "none of the per-edge intermediates reach device memory"
（`so2_conv.cu` 文件头注释）。实现方式也正是 §3 推导的形态：activation 驻留寄存器、
`__shfl_sync` 广播、权重经 LDS staging、在线 softmax、fp32 SIMT。

**所以 G2 段不需要我们再设计。** 但上游自己的标定对我们不利：

| checkpoint | 每层 per-edge FMA | fused vs Triton 组合 |
| --- | ---: | ---: |
| `mini` | 27.6 kFMA | **+43%** |
| `neo` | 112.6 kFMA | −4%（第二条 focus stream） |
| `air` | **225.3 kFMA** | **−9%** |

判据 `_MAX_PROFITABLE_LAYER_FMA = 65536` 就卡在两种符号之间（`so2_conv.py:855`）。

而**本 checkpoint 的形状正好是 `air`**：由 trace 可反解 `CF = 64`、`lmax = 3`
（`ROW = CF(3L+1) = 640`、`M0 = 256`、`M1 = 384`、`GATE = 192`，四个数与
`bmm [1,E,384]`、`bmm [1,E,256]`、`cat([E,256]⊕[E,384])`、`sigmoid [1,E,192]` 逐一对上），
于是 `m0² + m1² + CF·GATE = 65536 + 147456 + 12288 = 225 280`，**正是 `air` 那个 225.3 kFMA**。

三个直接后果：

1. 上游的 `_profitable()` 会**拒绝**在这个形状上启用 fused conv，`DP_CUDA_INFER=2` 等于白设；
2. 判据里还有一层设备缩放 `_device_ridge_scale()`（DCU 的 ridge 约 24–43，参考件是 73.2），
   缩放后预算只有 2–4 万 FMA，比 225 kFMA 小一个量级 —— 更加拒绝；
3. 所以 §2/§5 里"G2 段 megakernel −270…−320 ms"必须降级：那个数是**相对稠密参考路径**的，
   而上游只在 `air` 形状上验证过 fused **比 Triton 组合还慢 9%**。真正的比值是
   `(fused/Triton) × (Triton/稠密)`，第二个因子我们没有数据（Triton 在本机编译不了）。

**结论**：G2 的正确动作不是写 kernel，而是先测一个比值。具体做法：把
`so2_conv_fwd_c64_l3.cu` 单特化 hipify 出来，拿一次真实 conv 与稠密组合做 A/B；
若 < 0.6 再谈移植，否则 C7 直接砍掉。

## 5. 建议的推进顺序

| 档 | 内容 | 工作量 | 收益 | 风险 |
| --- | --- | --- | --- | --- |
| 一 | persistent 单算子：`sum [E,10,1,64]` 换 lane 映射（24.9 ms → ~3 ms）、readout 组（63.9 → ~40）、G5 反向 | 1–2 周 | −40…−55 ms | 低，纯换调度 |
| 二 | **先测 G2 的比值**：hipify `so2_conv_fwd_c64_l3.cu` 单特化，与稠密组合做一次 conv 的 A/B | 1 周 | 决定 C7 生死 | 低 |
| 二′ | 若比值 < 0.6：移植 fwd 特化（不重写算法） | 2–3 周 | 按实测折算 | 中，`.cu` → HIP + CMake 加 ROCm 分支 |
| 三 | 若比值 < 0.6：移植 bwd 特化（416.2 ms / 218.2 GB） | 3–4 周 | 按实测折算 | 高，`cuda_pipeline` / `__shfl_sync` / 无 `mma` |

第一档不受影响（纯换调度，约 −40…−55 ms）。第二/三档的收益一律改成"按实测折算"：
在本 checkpoint（`air`）上，上游的实测符号是**负**的，必须先拿到
`fused/Triton` 与 `Triton/稠密` 两个比值，否则任何数字都是猜的。

## 6. 判据：一个段值不值得 megakernel

1. `段字节 / 630 GB/s` 要明显小于实测时间（G2：470.6 vs 580 → 只靠带宽只有 19% 空间，
   必须靠减 pass；若某段已经贴着 roofline，megakernel 就是白做）。
2. 中间量能驻留：≤4 KB/边，且没有跨 block 依赖。
3. 段内 op 作用在同一批张量上（G2 满足；G3 的 848 次 launch 虽然多，但只有 12 ms）。

任一条不满足，就退回"persistent 单算子 + 减少 pass"的老路。

## 7. 风险清单

- **训练路径**：G2 的融合核只允许在 `not create_graph` 下启用，否则 DeepMD 的二阶导图会断。
- **精度**：10×10 mixing 的累加顺序变化会带来 fp32 末位差，需与 ATen 参考对齐
  （沿用 `tests/csrc/accuracy` 的 parity 框架，rtol 1e-5 / atol 1e-6）。
- **寄存器压力**：16 个槽只是起点，mixing 的累加还要 10 个；一旦 spill 到 scratch，
  收益会被吃掉，先按 1 边/wave 测通再谈 2 边/wave。
- **接缝**：段级融合需要在 `so2.py` 的 message 路径上加一个钩子（类似 `_segment_softmax_fn`），
  和 C1 一样面临"fork DeepMD 方法体"的问题，最好让上游加一行。
