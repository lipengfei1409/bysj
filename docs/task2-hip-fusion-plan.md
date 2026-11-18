# 任务二：基于 3000 原子 profile 的 HIP 融合算子方案

本文回答一个问题：**当前 3000 原子单点推理（能量 + 力）的 1218.6 ms 设备时间，应该融合哪些算子、为什么、怎么落地、能拿到多少**。
所有数字都来自仓库根的 `dpa4_inference_trace.json` / `dpa4_inference_profile_summary.json`，复现方式见 §8。

配套代码：`csrc/segment_sum.cpp`、`csrc/segment_softmax.cpp`、`csrc/force_assembly.cpp`，
Python 封装 `python/kernels/segment.py`，torch 参考实现 `python/reference/segment.py`，
主机测试 `tests/python/test_segment_ops_reference.py`，设备测试 `tests/csrc/accuracy/test_segment_ops_accuracy.py` 与
`tests/csrc/latency/test_segment_ops_latency.py`。

---

## 1. 结论

| # | 结论 | 依据 |
| --- | --- | --- |
| 1.1 | 单次推理 1218.6 ms / 2614 次 launch。**64%（780.4 ms / 1573 次 launch）在计算力的 autograd 反向**，只有 436.2 ms 在 descriptor 前向。 | §3.1 |
| 1.1b | **74.9%（913.2 ms / 1215 次 launch）的工作作用在 edge-major 张量 `[1,E,·,64]` 上**（连续同级 run 统计），node-major `[3000,·]` 只有 **7.7%（93.8 ms）**，gather/index 混合 run 5.2%（63.8 ms），其余 12.1% 是重排后的 bmm 形状。**融合的粒度应该按边流切，而不是按原子切。** | §10 形状分类 |
| 1.2 | 三个池子三种瓶颈，不能用同一招：**逐元素流 587.1 ms（三入一出的大张量已跑到 630–650 GB/s，只能减少 pass 数）**；**index/scatter 197.7 ms（只有 177 GB/s，卡在原子上，改 CSR 就能拿 3.5 倍）**；**GEMM 344.1 ms（8.1 TFLOP/s，是同卡 rocBLAS 13.7 TFLOP/s 的 59%，卡在形状与 epilogue）**。 | §3.2、§4 |
| 1.3 | 先做的三个算子 = **分段归约（fill+index_add+mul 三合一）**、**门控分段 softmax**、**力/维里装配（两次 CSR 遍历）**。可寻址池合计约 175 ms，全部是原子/多次 pass 造成的，改写不依赖新硬件特性；同时它们把 CSR 分组与段归约这套地基建好。 | §5 C1–C3 |
| 1.4 | 单项最大的是 **边流广播逐元素链融合**（池 587 ms，可省 180–220 ms）；**SO(2) 消息链融合**次之（−60…−80 ms），需先用上任务二的 GEMM epilogue。另有与 `bmm` 捆在一起的 `fill_`/`copy_`/`cat` prologue/epilogue（top 链里占 6 条，约 100 ms 级），和 C5 是同一件事的两半。 | §5 C4–C5、§10 |
| 1.5 | 预期：只做 C1–C3 → **约 1090 ms（1.12×）**；C1–C5 全做 → **约 820 ms（1.49×）**。GEMM 的 344 ms 不动；非 GEMM 部分从 874.5 ms 压到约 460 ms。 | §7 |

一句话：**这一版的加速空间不在"把 kernel 写快"，而在"把 3–5 个 pass 合成 1 个 pass、把原子换成 CSR 段"；GEMM 那一块留给 DMK 的 M 系列。**

---

## 2. 数据口径

| 项 | 值 |
| --- | --- |
| trace | `dpa4_inference_trace.json`（18 MB，kineto），summary 同目录 |
| 设备 | `K500SM_AI`（gfx928，wave64，120 CU，64 KB LDS），DTK 26.04，torch 2.9.0 |
| 输入 | 3000 原子水盒子，N=3000，**E=181838 条边**（约 60 边/原子） |
| 模型 | DPA4/SeZM legacy `pt` 后端，**3 个 interaction block**，通道 C=64，边张量 `[1,E,S,64]`（S=9/10） |
| 计时 | 设备 kernel 时间之和，不含 profiler 开销与 `.cpu()` 同步 |
| 关联方法 | `cuda_runtime` 事件的 `correlation` → kernel 事件取时长，`External id` → `cpu_op` 取算子名与 `Input Dims`；因此每次 launch 都能归到具体 aten 算子和形状 |

校验：2614 次 launch 的时长合计 1218.584 ms；前向 943 次 / 436.210 ms、反向 1573 次 / 780.375 ms、
其余 98 次 / 2.000 ms，三者相加为 1218.584 ms，与总时长 1218.5845 ms 一致（分区不重不漏，尾差为舍入）。
这五个数被 `scan_fusion_patterns.py` 固化成 `reference_checks`，任何一次重跑都会自动比对。

> `Input Dims` 是 profiler 记录的调用现场形状，个别条目会带 view/广播信息（例如 `index_add_` 的 index 维度）。
> 本文只在能自证的地方引用具体形状，语义解释以 DeepMD 源码为准。

---

## 3. 时间去哪了

### 3.1 按模型区域

| 区域 | 设备时间 | launch | 占比 |
| --- | --- | --- | --- |
| descriptor 前向（含 3 个 block） | 436.2 ms | 943 | 35.8% |
| **autograd 反向（`edge_energy_deriv`）** | **780.4 ms** | **1573** | **64.0%** |
| fitting / 其它 | 2.0 ms | 98 | 0.2% |

反向是主角。原因是能量对标量、力对坐标求导时，每个前向 kernel 平均产生约 1.7 个反向 kernel，
而这 1.7 个基本都是 elementwise / scatter —— **ATen 只能一个 kernel 表达一个算子，无法把 `grad = grad + a*b` 这种链条合并**。

### 3.2 按算子族

下表由 `tests/deepmd/profile/scan_fusion_patterns.py` 从 trace 独立生成（`docs/fusion-patterns.json` 的 `families`）：

| 族 | 设备时间 | launch | 占比 | 主要算子（时间 / 调用数） |
| --- | --- | --- | --- | --- |
| elementwise | 381.4 ms | 1390 | 31.3% | `mul` 208.1/626、`add_` 108.3/211、`add` 40.9/149 |
| memcpy | 130.3 ms | 318 | 10.7% | `copy_` 130.3/318 |
| init | 75.4 ms | 239 | 6.2% | `fill_` 75.4/235 |
| index | 197.7 ms | 198 | 16.2% | `index_add_` 92.9/39、`gather` 36.7/52、`index_select` 35.7/44、`_index_put_impl_` 19.5/9、`index` 8.2/13 |
| gemm | 344.1 ms | 292 | 28.2% | 289 次 `bmm`/`mm`，合计 2.78 TFLOP |
| reduce | 61.2 ms | 115 | 5.0% | `sum` 60.0/93、`mean` 1.1/15 |
| cat | 28.5 ms | 62 | 2.3% | `cat` 28.5/62 |

前三个族（elementwise + memcpy + init = **587.1 ms，1947 次 launch**）在本文里合成一个"逐元素流"池：
它们都是同一种访存模式的重复，区别只在 ATen 用哪个 kernel 实现。
index 族的 197.7 ms 与 gemm 族的 344.1 ms 各自单独成池。

### 3.3 单点热点（op + 形状，按时间排序）

| 时间 | 次数 | 算子 | 形状 |
| --- | --- | --- | --- |
| 85.0 ms | 24 | `bmm` | `[1,181838,384] × [1,384,384]` |
| 62.5 ms | 36 | `copy_` | `[1,181838,9,64]` |
| 59.7 ms | 27 | `add_` | `[1,181838,10,64]` |
| 52.5 ms | 27 | `mul` | `[1,181838,9,64]` |
| 40.0 ms | 24 | `bmm` | `[1,181838,256] × [1,256,256]` |
| 32.8 ms | 15 | `add` | `[1,181838,10,64]` |
| 32.7 ms | 9 | `index_add_` | `[1,181838,3,64]` |
| 27.7 ms | 6 | `bmm` | `[181838,10,10] × [181838,10,64]` |
| 26.6 ms | 12 | `add_` | `[1,181838,640]` |
| 26.0 ms | 27 | `fill_` | `[1,181838,10,64]` |
| 24.9 ms | 3 | `sum` | `[181838,10,1,64]` |
| 22.3 ms | 12 | `cat` | `[1,181838,256] ++ [1,181838,384]` |
| 22.1 ms | 9 | `mul` | `[84,181838]` |
| 20.7 ms | 1 | `index_add_` | `[181838,16,16] ← [181838,16,10]` |
| 20.5 ms | 9 | `index_select` | `[1,181838,3,64]` |
| 12.6 ms | 3 | `index_add_` | `[3000,16,64] ← [181838,16,64]` |

---

## 4. 三个 regime：先判瓶颈，再决定改什么

把上面每个热点换算成实测带宽 / 算力，瓶颈立刻分开（峰值参照：同卡 rocBLAS SGEMM 实测 13.7 TFLOP/s；
elementwise 实测可达 632 GB/s）：

| regime | 代表 op | 实测 | 同量纲可达 | 距离 | 唯一有效的做法 |
| --- | --- | --- | --- | --- | --- |
| 访存受限 | `add_ [1,E,10,64]`（三入一出） | 632 GB/s（1.40 GB / 2.21 ms） | ~640–700 GB/s | 已到顶 | **减少 pass 数**（融合），换 kernel 写法无用 |
| | `copy_ [1,E,9,64]`（一入一出） | 482 GB/s（0.838 GB / 1.74 ms） | ~640 GB/s | 1.3× | 同上；`copy_` 还有一部分是纯搬运，可直接消除 |
| 原子受限 | `index_add_ [3000,16,64]` | 177 GB/s（0.745 GB / 4.20 ms） | ~630 GB/s | **3.5×** | **改 CSR 段归约**，去掉行原子 |
| 形状/epilogue 受限 | `bmm [1,E,384]×[384,384]` | 8.1 TFLOP/s | 13.7 TFLOP/s | 1.7× | GEMM epilogue + tile（任务二 M 系列） |

以大张量三入一出为例（`add_`/`mul` 家族），实测落在 630–650 GB/s 的窄带里，
说明这批 kernel 已经贴着这张卡对 float32 三流式的持续带宽；而 `copy_`/`fill_` 这类
两流/单流的只有 350–480 GB/s（写带宽与小张量 launch 双重折损）。

这条判据直接决定了候选的优先级：**原子受限的池子（197.7 ms）投入产出最高**，因为不需要新硬件特性就能拿 3 倍；
**逐元素流池子（587.1 ms）只能靠 C4 那种"多条链一次读完"的融合**；
GEMM 池子（344.1 ms）单独留给 DMK。

> 为什么不是"把 elementwise kernel 再优化一下"：`add_ [1,E,10,64]` 单次 2.21 ms 搬 1.40 GB，
> 已经接近这张卡的持续带宽；再快的 kernel 也只是把 632 GB/s 提到 700 GB/s（+10%）。
> 而把 3 个 pass 合成 1 个是 −66% 的访存量。这就是本文所有候选的立论基础。

---

## 5. 候选融合算子

优先级按"池子大小 × 瓶颈可解程度 ÷ 实现成本"排序。C1–C3 已实现（代码在仓库里），C4–C5 给出设计。

### C1 分段归约：`fill_` + `index_add_` + 归一化 → 1 次 CSR 遍历 ★已实现

**证据（trace）**

```
fill_(out, 0)                235 次 / 75.4 ms    ← 其中 [3000, C] 这类小张量单次 0.3 ms，纯属 launch 开销
index_add_(out, dst, src)     39 次 / 92.9 ms    ← 91.7 ms 落在 indexFuncSmallIndex/indexFuncLargeIndex(ReduceAdd)
mul_(out, inverse_degree)     …                  ← 又一次全量读+写
```

`index_add_ [3000,16,64] ← [181838,16,64]` 是 4.20 ms 搬 0.745 GB = **177 GB/s**，
而同一张卡上纯 gather 能跑 630 GB/s —— 差在 60 个边/原子撞同一行的行内原子。
DeepMD 自己的 Triton 实现（`deepmd/pt_expt/kernels/triton/sezm/force_assembly.py`）对同一现象的原话是
"row-atomic scatters serialize on the colliding edges of each atom"，因此它也用 CSR 段归约。

**数学**：`out[n,c] = s[n] · Σ_{dst(e)=n} w[e]·src[e,c]`

**接口**（已实现）：`segment_sum(src, dst, offsets, edge_scale=None, node_scale=None)`

- `src (E,C) fp32`，`dst (E,) int64` 非降，`offsets (N+1,)` 是 `dst` 的 exclusive prefix sum；
- `edge_scale (E,)` 逐边权、`node_scale (N,)` 逐原子后乘；两者都可为 `None`；
- 反向：`grad_src[e] = grad_out[dst[e]]·w[e]·s[dst[e]]`，`grad_node_scale[n] = Σ_c grad_out[n,c]·prefix[n,c]`（原地重算，不存中间量）。
- 分组只需在构图时做一次：`offsets = segment_offsets(dst, N)`，之后 3 个 block 与反向复用。

**为什么值得做**：一次遍历替代三次 launch；累加器在寄存器里，**零原子**；归一化折进 store；
小张量的 `fill_`（单次 0.3 ms，init 池共 75.4 ms）直接归零。

**开发过程**

1. 先在 `python/reference/segment.py` 用 ATen 链写"可执行规格"，`tests/python/test_segment_ops_reference.py`
   用独立 for 循环对照，并把闭式反向对 autograd 校验（已通过，14 个用例）；
2. `csrc/segment_sum.cpp`：一个 block 一个目标段，`blockDim = min(roundup(C,64),256)`，C%4==0 走 `float4` 分支；
3. 设备精度：`tests/csrc/accuracy/test_segment_ops_accuracy.py` 覆盖 C=1/3/64/640（标量与向量两条路径）与梯度；
4. 延迟 A/B：`tests/csrc/latency/test_segment_ops_latency.py` 用 trace 的真实尺寸跑 fused vs `fill_+index_add_+mul_`。

**预期结果**

| 项 | 现在 | 目标 | 依据 |
| --- | --- | --- | --- |
| `index_add_` 的 node-major 子集（目标 `[3000,·]`） | ~40 ms | ~12 ms | 177 → 630 GB/s，3.5× |
| 这些小张量的 `fill_` | ~10 ms | 0 | 融合进累加器 |
| 归约后的广播 `mul_`/`mul` | ~5–10 ms | 0 | 折进 store |
| 合计 | ~55 ms | ~12 ms | **−35…−45 ms** |

口径说明：`index_add_` 家族总量是 92.9 ms，但其中 `[1,E,3,64]`（32.7 ms）属于 C3、
`[E,16,16]`（20.7 ms）属于 C5，本候选只吃"目标是 `[3000,·]`"的那部分。
单看收益不算大（−35…−45 ms），但它是 C3/C5 复用的基础设施：CSR 分组、段归约、
以及 `grad_node_scale` 的"重算 prefix"模式都在这里一次性建好。
**硬天花板**：整份 trace 里 node-major 的 run 只占 93.8 ms（§10），所以"按原子归约"这条路无论怎么写都拿不到 100 ms 以上，
C1 的定位就是"低成本、可复用、顺带校准判据"，不是主力收益。

**风险**：段过长（某原子边数上百）时单 block 串行度上升 —— 用 `blockDim` 按 C 拉满缓解，
实测若某段 >2048 边需改成多 block + 两级归约（本模型 60 边/原子，暂不需要）。

---

### C2 门控分段 softmax：`fill_` + 多次 `index_add_` + `exp/div` → 1 个段内核 ★已实现

**证据（trace）**：每个 block 都有一组 `[3000,256]` 的 `fill_` + `index_add_`，
加上 `reduce_kernel<512,1>` 70 次 / 28.9 ms、`sum [181838,10,1,64]` 3 次 / 24.9 ms、
以及把 envelope 门控、null mass 折进去的那几条 `mul [1,181838,4,1] × [1,181838,1,64]`。
这些合起来是"一次归一化、五六个 launch、每个都是一整趟 `(E,C)`"。

**数学**（与 `triton/sezm/segment_softmax.py` 的公式一致，逐通道）：

```
l_e = logit_e + 2·ln env_e   (env_e > 0，否则该边不参与)
m   = max(l_0, max_e l_e)
α_e = exp(l_e − m) / D,  α_0 = exp(l_0 − m) / D,  D = exp(l_0 − m) + Σ_e exp(l_e − m)
```

一阶反向（给定 `ᾱ`，**不需要再读一遍 logits**）：

```
S = Σ_e α_e ᾱ_e + α_0 ᾱ_0
∂l_e = α_e (ᾱ_e − S),   ∂l_0 = α_0 (ᾱ_0 − S),   ∂env_e = (Σ_c ∂l_e) · 2/env_e
```

**接口**（已实现）：`segment_softmax(logit, env, offsets, null_logit=None) -> (alpha, null_alpha)`

**为什么值得做**：最大值与分母都留在寄存器；null mass 不再单独物化；反向复用 `α`，
省掉"重读 logits + 重算 exp"的一整趟；`(E,256)` 张量少读 2 次 ≈ 少搬 0.37 GB。

**开发过程**

1. 从 `triton/sezm/segment_softmax.py` 的三段式（前向 / 一阶 / 二阶）里只取前向 + 一阶，
   把 `l_e = logit + 2·ln env` 收成一个 `effective_logit` 内联函数，`env ≤ 0` 直接 skip；
2. 参考实现用 `scatter_reduce(amax)` + `index_add` 复刻 ATen 链；主机测试断言
   "每段每通道 `Σ_e α_e + α_0 = 1`"、"`env ≤ 0` 的边质量为 0"、"无 null 时仍归一"；
3. 闭式反向对 autograd 校验（`SegmentSoftmaxReferenceTests::test_closed_form_backward_matches_autograd`）；
4. 设备侧：一个 block 一段、`blockDim = channel_threads(C)`；`grad_env` 单独一遍，每边一个 block 做通道归约；
5. 延迟 A/B：C=64/256、E=181838（`tests/csrc/latency/test_segment_ops_latency.py`）。

**预期结果**：池 ~25 ms（`fill_[3000,256]` + `index_add_[3000,256]` + `reduce_kernel<512,1>` 里属于软最大/求和的部分）
→ ~8 ms（**−15…−20 ms**）；归一化从 6 个 launch 降到 1 个，`(E,256)` 的 logits 少读 2 趟。

**风险**：数值路径与 torch 的 `scatter_reduce(amax)` 不完全一致（同样的最大值，但累加顺序不同），
只承诺 rtol 2e-5、不承诺 bit-exact —— 与 DeepMD 现有 Triton 路径的容差口径一致。

---

### C3 力/维里装配：两次原子散射 + `(E,9)` 物化 → 两次 CSR 遍历 ★已实现

**证据（trace）**：`index_add_ [1,181838,3,64]` 9 次 / 32.7 ms、`_index_put_impl_` 6 次 / 13.9 ms、
`scatter_gather` 29 次 / 18.0 ms、`indexing_backward` 3+1 次 / 19.3 ms，
再加上把 `g_e ⊗ v_e` 物化成 `(E,9)` 的那几条 `mul`/`copy_`。合计约 90–110 ms。

**数学**（同 `triton/sezm/force_assembly.py`）：

```
F_k = Σ_{dst(e)=k} g_e − Σ_{src(e)=k} g_e
W_k = Σ_{src(e)=k} (−g_e ⊗ edge_vec_e)     # 9 个分量，行主序
```

**接口**（已实现）：`force_assembly(grad_edge, edge_vec, src, dst, dst_offsets, src_offsets=None, src_order=None)`

- 自然边序按 dst 分组（DeepMD 布局），dst 侧直接用 `dst_offsets` 寻址；
- src 侧需要显式置换 `src_order` + `src_offsets`，未给时由 `group_by_source(src, N)` 一次 `argsort` 生成（可缓存）；
- 一个 block 带 5 个原子（60 lane 填满一个 wave64），lane 0–2 算力、3–11 算维里，**double 累加、零原子、无中间张量**。

**为什么值得做**：把"两次原子散射 + 一次 `(E,9)` 物化"换成"两次只读遍历"，
且累加顺序确定（对力的可复现性有好处）。

**开发过程**

1. 从 `triton/sezm/force_assembly.py` 取两段式结构与"source pass 重算外积、全量归到源端"的原子维里口径；
2. 先钉布局契约：自然边序按 dst 分组（dst 侧零拷贝寻址），src 侧用 `src_order` 置换 +
   `src_offsets`，由 `group_by_source` 一次 `argsort` 生成并可缓存；
3. 参考实现用两次 `index_add_` + `(E,9)` 物化；主机测试额外验守恒律 `Σ_k F_k = 0`（每边 +g/−g 相消）；
4. 设备核：5 原子/block、12 lane/原子、double 累加，无原子无中间张量；
5. 延迟 A/B：E=181838，`argsort` 单独计时以确认它只值一次图构建。

**预期结果**：池 ~95 ms（`index_add_[1,E,3,64]` 32.7 + `_index_put_impl_` 13.9 + `scatter_gather` 18.0 +
`indexing_backward` 19.3 + `(E,9)` 物化约 10）→ ~25 ms（**−65…−75 ms**）。
这 95 ms 全部落在 edge-major run 里（§10），是"按边流融合"这条主线上的第一块。

**风险**：`src_order` 的 `argsort`（E=181838，约 1.5 ms）必须只做一次并复用；推理专用，
若需要 `create_graph` 的二阶力路径，保留原 ATen 路径。

---

### C4 边流广播逐元素链融合（设计，未实现）

**证据**：逐元素流池 587.1 ms（elementwise 381.4 + memcpy 130.3 + init 75.4），其中 `[1,E,·,64]` 形状的 `add_/add/mul/copy_/fill_`
就是"残差 + 门控 + 注意力加权 + 反向的广播乘加"。典型三元组：
`add_ [1,E,10,64]`(59.7) → `add_ [1,E,640]`(26.6) → `mul [1,E,9,64]`(52.5) → `copy_ [1,E,9,64]`(62.5)。

**做法**：一个"描述符驱动"的逐元素内核，一次读入 k 个输入、按项表（指针 + 步长 + 系数 + 广播轴）累加，
写入 1 个输出，末尾带可选激活（silu/sigmoid，与 `triton/sezm/gated_activation.py` 的
`act = [s·σ(s), z_r·σ(q)_{g(r)}]` 对齐）。由于每个 pass 已经在 632 GB/s，
**收益完全来自 pass 数从 k 降到 1**：k=3 时 −66% 访存。

**预期结果**：池 587.1 ms → 370–410 ms（**−180…−220 ms**）。

**开发过程**

1. 用 §10 的 n-gram 选出重复出现的 3–4 元链（当前 top 链每条重复 9 次），按"输入个数 + 广播轴"
   归成 3–4 个模板，而不是做通用图编译器；
2. 先写"项表 → 一次 pass"的内核（项 = 指针 + 步长 + 系数 + 广播轴 + 可选激活），
   用参考实现（逐项 eager 相加）做主机 A/B；
3. 设备侧按 `[E, 9/10, 64]` 布局做 `float4` 访存，先覆盖 backward 里最常见的
   `accumulate + gate + residual` 三种形态；
4. gate 的判据不是"kernel 更快"，而是"pass 数从 k 降到 1"——所以在 trace 上直接数 pass、在设备上量字节。

---

### C5 SO(2) 消息链融合（设计，未实现）

**证据**：每个 block 的 `index_select [1,E,3,64]`(20.5) → `mul [1,E,9,64]`(17.3) →
`copy_ [1,E,9,64]`(15.6) → `add [1,E,10,64]`(26.2) → `bmm [1,E,384]`(42.1) → `cat`(22.3)，
3 个 block 合计约 120–150 ms；反向还要再来一遍并多出 `84×E` 的 radial 调制（22.1 ms）。

**做法**：按 `triton/sezm/so2_value_path.py` 的思路 —— rotate-to-local + radial 调制 + block GEMM +
rotate-back + 目标段散射，**中间旋转结果不落 HBM**（Triton 版把旋转保持在寄存器里，
并用 CSR `segment_sum` 取代 `index_add_` 做节点梯度归约）。落地上分两步：
先把 C1 的段归约复用为节点梯度；再把 GEMM 换成 DMK 的分块 GEMM 并做 epilogue 融合
（这正是任务二 M 系列的工作，所以排在第二阶段）。

**预期结果**：~130 ms → ~60 ms（**−60…−80 ms**）。
**风险**：与任务二的 GEMM 进度耦合；DUMMA `m16n16k4/k8` 的分块形状要和 `S=9/10`、C=64 对齐。

---

## 6. 已交付代码

| 文件 | 内容 |
| --- | --- |
| `csrc/segment_sum.cpp` | C1：CSR 段归约（前向 + `grad_src` + `grad_node_scale`），标量/`float4` 两条路径 |
| `csrc/segment_softmax.cpp` | C2：门控分段 softmax（前向 + 一阶反向 + `grad_env`） |
| `csrc/force_assembly.cpp` | C3：双 CSR 力/维里装配，5 原子/block，double 累加 |
| `csrc/ops.h`、`csrc/bindings.cpp`、`setup.py` | 入口声明、pybind 绑定、默认编译列表 |
| `python/kernels/segment.py` | 封装 + `autograd.Function` + `segment_offsets` / `group_by_source` |
| `python/reference/segment.py` | ATen 参考实现（可执行规格） |
| `tests/python/test_segment_ops_reference.py` | 14 个主机测试：参考实现 vs 独立循环，**闭式反向 vs autograd** |
| `tests/csrc/accuracy/test_segment_ops_accuracy.py` | 设备精度/梯度测试（无 DCU 自动 skip） |
| `tests/csrc/latency/test_segment_ops_latency.py` | 真实尺寸 A/B 延迟，输出 JSON（`DPA4_SEGMENT_LATENCY_JSON`） |

设计上的三条硬约束（都已在代码里落实）：

1. **不写 zero-init**：前向 `torch::empty` + 寄存器累加，输出每个元素都被写到；
2. **不用行原子**：所有归约按 CSR 段在 block 内完成，结果与累加顺序无关；
3. **反向不重算前向**：softmax 反向只读 `α`；`segment_sum` 的 `grad_node_scale` 才重算 prefix，且是可选分支。

---

## 7. 预期收益与验收 gate

| 阶段 | 动作 | 增量 | 累计设备时间 | 相对基线 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 基线 | — | — | 1218.6 ms | 1.00× | 实测 |
| 1 | C1 分段归约 | −35…−45 ms | — | — | 已实现，待设备实测 |
| 2 | C3 力装配 | −65…−75 ms | — | — | 已实现，待设备实测 |
| 3 | C2 分段 softmax | −15…−20 ms | **1080–1100 ms** | **1.11–1.13×** | 已实现，待设备实测 |
| 4 | C4 逐元素链 | −180…−220 ms | 880–900 ms | 1.35–1.38× | 设计 |
| 5 | C5 SO(2) 链 | −60…−80 ms | **810–830 ms** | **1.47–1.50×** | 设计（依赖 M 系列） |

GEMM 的 344.1 ms 不在上表内；非 GEMM 部分从 874.5 ms 压到约 460 ms 是这套方案的天花板
（再往下就要动 GEMM 的 epilogue，那是任务二 M 系列的事）。

收益结构的直白版本：**已实现的三个算子合计只值 −115…−140 ms（≈1.12×），真正的大头是 C4 的 −200 ms**。
把 C1–C3 排在第一阶段，是因为它们成本低、能立刻校准"177 → 630 GB/s"这条判据，
并且 C3/C5 都要复用它们的 CSR 分组与段归约；不是因为它们能单独改变数量级。

**验收 gate（建议写进周报）**

| gate | 判据 | 测量方式 |
| --- | --- | --- |
| G-A 精度 | 三个算子 vs 参考实现 rtol ≤ 2e-5（梯度 ≤ 2e-4），空段/负 envelope/单通道等边界全覆盖 | `tests/csrc/accuracy` |
| G-B 单算子 | C1 在 `[3000,16,64]←[181838,16,64]` 上 ≥ 3× `fill_+index_add_+mul_`；C3 ≥ 3× ATen 链 | `tests/csrc/latency` 的 JSON |
| G-C 端到端 | 3000 原子单点推理设备时间 ≤ 1100 ms（保守取 −115 ms） | `test_dpa4_inference_profile.py` 同一协议 |
| G-D 数值回归 | 能量/力与基线逐元素对比：能量 rtol 1e-6，力 rtol 1e-4（原子顺序不同，按容差） | profile CLI 输出 |

---

## 8. 复现命令

```bash
# 主机（无需 DCU）：参考实现、闭式反向、构建契约
cd myproject
PYTHONPATH=python python3 -m unittest discover -s tests/python -v

# DCU：编译 + 设备精度
source /opt/dtk/env.sh
cd myproject && MAX_JOBS=4 python3 setup.py build_ext --inplace
PYTHONPATH=python python3 -m unittest discover -s tests/csrc/accuracy -p 'test_segment_ops*' -v

# DCU：A/B 延迟（真实尺寸 3000 原子 / 181838 边）
DPA4_RUN_SEGMENT_LATENCY=1 DPA4_SEGMENT_LATENCY_JSON=/tmp/segment_latency.json \
  PYTHONPATH=python python3 -m unittest discover -s tests/csrc/latency -p 'test_segment_ops*' -v

# 端到端（同一协议，便于和 1218.6 ms 对齐）
DP_TRITON_INFER=0 python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
  --model-file <ckpt> --input-npz <npz> --fixed-input --device cuda:0 --output-dir /tmp/prof
```

---

## 9. 风险与开放问题

| # | 风险 | 影响 | 处置 |
| --- | --- | --- | --- |
| R1 | legacy `pt` 后端的边序不保证按 dst 分组 | C1/C2/C3 都要先做一次分组（一次 `argsort`/计数排序，约 1.5 ms，可缓存） | 用 `segment_offsets` 校验；adapter 里缓存到图或输入不变为止 |
| R2 | 段长分布尾部（某些原子边数远大于均值） | 单 block 串行 | 先测 P99 段长；超过 2048 时切多 block + 两级归约 |
| R3 | 原子加法换成有序累加，结果与基线不完全一致 | 力的最低位不同 | G-D 用 rtol 1e-4；确定性反而是加分项 |
| R4 | softmax 的 max 取法与 torch 路径不同 | 极小概率在 -inf 边界上不同 | 显式测 `env ≤ 0`、全屏蔽段、`null_logit = -inf` 三种边界 |
| R5 | C4 的"项表"若做成通用图编译器，工期不可控 | 拖垮周计划 | 只按 n-gram top-15 写死变体 |
| R6 | C5 依赖 DMK GEMM 的 epilogue | 排期耦合 | 先做 C1 复用的节点梯度部分 |

开放问题（需要设备实测才能定）：

1. C=64 时一个 block 只有 64 线程（一个 wave），3000 个 block 的占用率是否够——
   若不够，改成"一个 block 多个段 + 每段一个 wave"的二级布局；
2. `float4` 分支在 C=640 时每线程 2.5 组，是否有对齐收益；
3. 反向 `grad_env` 的"每边一个 block"（E=181838 个 block）是否比"每段一个 block"更好。

---

## 10. 附录：相邻算子模式扫描

`tests/deepmd/profile/scan_fusion_patterns.py` 把 trace 的 launch 序列按"cpu_op 名 + 形状"切成 n-gram，
统计每条链的出现次数、总时间、可寻址字节与在 630 GB/s 下的下界，用来挑 C4 的 top 链。
命令：

```bash
python3 tests/deepmd/profile/scan_fusion_patterns.py \
  --trace ../dpa4_inference_trace.json --top 25 --max-len 5 \
  --json docs/fusion-patterns.json
```

扫描结果（`docs/fusion-patterns.json`，均可复算）：

| 校验项 | 扫描值 | 期望 | 一致 |
| --- | --- | --- | --- |
| kernel 事件数 | 2614 | 2614 | ✓ |
| 设备时间和 | 1218.584 ms | 1218.584 ms | ✓ |
| 前向（launches / ms） | 943 / 436.210 | 943 / 436.2 | ✓ |
| 反向（launches / ms） | 1573 / 780.375 | 1573 / 780.4 | ✓ |
| 形状哨兵（3000 / 181838） | 1102 / 1296 次 | 原子数 / 边数出现次数 | — |

#### 形状分类：边流占四分之三

把每次 launch 按 `Input Dims` 里是否含 `[1,181838,…]`/`[181838,…]`（edge）或 `[3000,…]`（node）分类，
再把相邻同级 launch 合成 540 个极大 run：

| 类别 | 设备时间 | 占比 | launch | 其中长度 ≥2 的纯链 |
| --- | --- | --- | --- | --- |
| **edge-major** | **913.20 ms** | **74.9%** | 1215 | 901.94 ms（最长 run 157） |
| node-major | 93.77 ms | 7.7% | 886 | 79.00 ms |
| gather/index 混合 | 63.79 ms | 5.2% | — | — |
| 重排后形状（`[192,1,192]`、`[1,152,48]×[1,48,576000]`、`[16,3000,1152]`） | 147.83 ms | 12.1% | — | — |

两个直接后果：

1. **node-major 全部加起来只有 93.8 ms** —— C1 这种"按原子归约"的算子天花板就在这里（见 §5 C1）；
2. **edge-major 的 913.2 ms 里，901.9 ms 处在长度 ≥2 的纯边链中** —— 这正是 C3/C4/C5 的作业面，
   也说明"一条边流、多个算子"的 megakernel 粒度是对的。

#### 最热的相邻链（top 8，n≥2）

| n | 次数 | 合计 | 链（简写） |
| --- | --- | --- | --- |
| 5 | 12 | 99.98 ms | `bmm[E,384]` → `fill_[E,640]` → `copy_[E,384]` → `bmm[E,256]` → `fill_[E,640]` |
| 4 | 12 | 88.43 ms | `bmm[E,384]` → `fill_[E,640]` → `copy_[E,384]` → `bmm[E,256]` |
| 5 | 9 | 86.25 ms | `add_[E,10,64]` → `bmm[E,384]` → `fill_[E,640]` → `copy_[E,384]` → `bmm[E,256]` |
| 5 | 12 | 84.57 ms | `copy_[192,1,192]×2` → `bmm[E,256]` → `bmm[E,384]` → `cat` |
| 3 | 12 | 84.33 ms | `bmm[E,256]` → `bmm[E,384]` → `cat` |
| 5 | 9 | 82.51 ms | `add_[E,10,64]` → `fill_[E,10,64]` → `copy_[E,64]` → `add_[E,10,64]` → `bmm[E,384]` |
| 5 | 9 | 76.73 ms | `index_select[E,3,64]` → `copy_[E,1,64]` → `mul[E,9,64]` → `copy_[E,9,64]` → `add[E,10,64]` |
| 5 | 9 | 75.06 ms | `copy_[E,9,64]` → `mul[E,9,64]` → `mul[E,9,64]` → `fill_[E,10,64]` → `copy_[E,9,64]` |

读法：**榜首几条是 bmm 与它前后那两三个全量 pass 捆在一起**（`fill_[E,640]` 是 GEMM 输出缓冲的零初始化/
清零，`copy_[E,384]` 是把 GEMM 结果搬进拼接缓冲）。这说明还有第四类融合没被 C1–C4 覆盖：
**GEMM 的 prologue/epilogue**（`fill_`/`copy_`/`cat` 与 `bmm` 绑在一起，合计约 100 ms 级）。
它和 C5 是同一件事的两半，都要求 GEMM 能接受"预置输入/直接写目标布局"，也就是 DMK 的分块 GEMM epilogue。

> 注意事项：不同 n 的窗口互相嵌套（同一 12 次出现会在 n=3/4/5 里各出现一次），
> 因此这些行的收益**不可相加**；工具里的 `saving_ms` 又按"每项 3 路流 × 最大 numel"估算，
> bmm 主导的链会偏乐观、逐元素链会偏保守（出现负值 = 已经快于该模型）。
> 这些数只用于**排序**，§7 的收益承诺是按手工核对的池子给的。

---

## 附：与 Triton 的关系（一句话）

本方案的三个候选与 `deepmd/pt_expt/kernels/triton/sezm/` 里的 `segment_softmax.py`、`force_assembly.py`、
`so2_value_path.py` **数学一致、分解一致**（CSR 段归约取代 `index_add_` 原子、null mass 折进 softmax、
旋转不落 HBM），区别只在实现层：那边是 Triton + `tl.dot`，这边是 HIP/DUMMA。
本机 Triton 的 `tl.dot` 路径因前端（LLVM 22）与后端 gcvm（LLVM 17）代际不匹配无法编译，
因此这批算子在 DCU 上只能走 HIP —— 这也是把它们做进 `csrc/` 而不是 Triton 的原因。
