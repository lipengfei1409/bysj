# 给 DPA4 做一条模型专用 persistent 调度

通用框架已经能跑（`csrc/dmk/` + `python/dmk/`），但“通用”本身不产生收益。这一页回答：
**如果只为 DPA4 这一个模型写调度，应该打哪里、怎么写、预期多少。**

## 1. 先看数据：收益在哪

来自 [`optimization-status.md`](optimization-status.md) 的实测分解（单步，DCU）：

| 段 | 时间 | 占比 | 性质 |
| --- | --- | --- | --- |
| `G2 槽位段`（C7：per-edge 小 bmm + `sum(dim=2)`） | 580.0 ms | 47.6% | 访存为主，融合版**已实测判负**（0.80×） |
| **E-major 逐元素链（C6）** | **488.1 ms** | **40.1%** | 同一条 3–4 kernel 链**重复 9 次** |
| readout 瘦 GEMM（C5） | 63.9 ms | 5.3% | K=48，算力空转 |
| GEMM + 残差（C3） | 8.1 TFLOP/s（roofline 13.7） | — | 已经不是瓶颈 |

结论很清楚：**持久化调度该打 C6 那条链，不是 GEMM。** 三个理由：

1. C6 是**逐元素**的，完全不需要矩阵单元 → 绕开 DUMMA 的全部风险；
2. “同一条 3–4 kernel 链重复 9 次”正是 megakernel 的靶心：9 次发射、9 次中间量落 HBM；
3. C7 的融合版已经判负（0.79–0.80×），说明**不是所有链都适合融合**——C6 的账要先算清再动手。

## 2. 账怎么算（这是判断依据，不是实测）

一条 `[E, k, 64]` 的 3-pass 链，如果 k=9、每元素 4 B：

| 项 | 现在 | 融合后 |
| --- | --- | --- |
| kernel 发射 | 3 次 × 9 条链 | **1 次 × 9 条链** |
| 中间量写 + 读 | 2 趟往返 × 中间量 | **0**（留在页里） |
| HBM 流量 | ≈ 5 趟（读入 + 3 写 + 2 读） | ≈ 1 趟读 + 1 趟写 |

也就是说理论下限是**现在的 2/5**：488 ms → 约 195 ms，**−290 ms/step**。这个数字是算术，不是实测；
真实值取决于每趟的实际带宽利用率（C6 的注释里记着有效带宽远低于峰值，说明还有额外浪费）。

## 3. 模型专用意味着什么

| 通用（现在） | 模型专用 |
| --- | --- |
| 指令流从 HBM 读，`make_op` 按 opcode 分派 | **顺序写死**在 kernel 里（`switch (stage)` 或模板展开） |
| controller 波负责取指/发布/页置换 | 不需要 controller，不需要页置换 |
| 一个 op 一类算子，谁都能加 | 只为这条链的 4 个阶段写代码 |
| 分片由 `plan_cus` 通用计算 | 分片按 `(E, k, 64)` 直接算，可离线预生成 |

省掉的部分恰好是通用框架里最难调的部分（页生命周期、信号量、指令解码）。**保留**的是它真正有用的
两块：8 KiB 页作为暂存、以及 `__launch_bounds__`/LDS 预算那一套。

## 4. 目录与文件

```
csrc/dmk/dpa4_chain.cpp     一个 kernel：9 条链共用一个静态调度
python/dmk/dpa4_plan.py     主机侧：从 (E, k, C, 链定义) 算出每 CU 的边块与阶段表
tests/python/test_dpa4_plan.py     分片正确性（主机）
tests/csrc/accuracy/test_dpa4_chain.py  与 edge_combine 路径逐位对拍（DCU）
```

kernel 形状（无指令解码、无双缓冲也能先跑）：

```cpp
__global__ void __launch_bounds__(kThreads) dpa4_chain(Dpa4Globals g) {
    for (stage = 0; stage < g.num_stages; ++stage) {     // 静态表，来自 HBM 常量
        load_terms(stage);      // 项的因子读进页
        __syncthreads();
        fmad_terms(stage);      // 一趟算完，写回页
        __syncthreads();
        store_once(stage);      // 只有最后一个 stage 写 HBM
    }
}
```

## 5. 第一步做什么（最小可判定的实验）

**只做一条链**，与现有 `edge_combine` 路径对拍 + 计时：

1. 用真实 `(E, k, 64)` 的随机张量，跑 `edge_combine`（已有算子）记时间；
2. 用 `dpa4_chain` 跑同一条链，逐位对拍数值；
3. 若 > 1.5×，再推广到 9 条链并接进 DeepMD；若 ≤ 1.2×，停下来重新算账
   （很可能这条链的主要成本不在发射，而在带宽利用率）。

这个实验一天内能出结果，而它决定的是**40% 的时间**。

## 6. 不建议做的事

| 事项 | 原因 |
| --- | --- |
| 继续优化持久化 GEMM | GEMM 已 8.1 TFLOP/s（roofline 13.7），`ratio` 天花板只有 −30…−35 ms；且打不过 rocBLAS |
| 把 C7 的 conv 再融合一次 | 已实测 0.80× |
| 先做全模型一张图 | 权重 MB 级、LDS 只有 64 KiB；单层单链才是可复用单元 |
