# C6 审计：E-major 逐元素链到底有没有可融的东西

C6 的前提是"同一条张量上有 3–5 个连续的逐元素 op，可以一趟读完、一趟写完"。
这个前提**可以量**，而且从家族总量看不出来：逐元素占 31.3% 设备时间，完全可能是
一堆**互相隔着 GEMM / gather / scatter 的孤立单点 op**——那种没有链可融。

工具：[`chain_runs.py`](../tests/deepmd/profile/chain_runs.py)（同形连续 run + `fill_→accumulate` 配对）。

```bash
python3 tests/deepmd/profile/chain_runs.py ../dpa4_inference_trace.json
```

## 结果（3000 原子 / 181838 边 / 2614 launch）

| run 长度 | 合计 ms | run 数 |
| ---: | ---: | ---: |
| 2 | 84.5 | 186 |
| 3 | 1.2 | 5 |
| 4 | 14.1 | 2 |
| 5 | 0.7 | 10 |
| 9 | 0.2 | 1 |

- 同形逐元素 run 合计 **100.7 ms**，其中**长度 ≥3 的只有 16.2 ms**；
- `fill_` 合计 75.4 ms，但**`fill_→accumulate` 配对为 0**（没有"先清零再累加"这种可以被写入替代的模式）。

最长的几条也都很小：`[84, 181838]` 上 `mul→mul→mul→mul` 14.1 ms，
`[1,E,9,64]` 上 9 条 `mul→mul` 合计约 35 ms。

## 结论：C6 按原设计拿不到东西

- 2-op run 融合的理论收益是"少写一遍"：2 读 2 写 → 2 读 1 写，省 1/3；
  对 84.5 ms 里能确定同张量的部分，上限约 **20–25 ms**，落到 kernel 上更少；
- 长度 ≥3 的 16.2 ms 不值得写一个项表驱动的 FMAD 模板；
- 488 ms 的逐元素池**不是链**：它是孤立的单点 op，两侧被 GEMM/gather/scatter 隔开。
  要减少这部分字节，只有两条路，都不在"逐元素链 kernel"这个框架里：
  1. **跨 GEMM 融合**（把 epilogue/prologue 吸进 GEMM）——属 megakernel / DMK 线；
  2. **减少被物化的值**（比如多余的梯度缓冲区、可以原地写的消费者）。

所以状态表里 C6 记为 ❌ 并给出实测依据，`-180…-220 ms` 的估算作废。

## 替代路线（都有具体数字，且不依赖设备 ridge）

| 路线 | 池子 | 做法 | 预期 |
| --- | ---: | --- | --- |
| **C1b** gate 展开的反向 | 35.3 ms（`index_add_ [1,E,3,64] ⬅ [1,E,9,64]` 32.7 + `fill_` 2.6） | 逐边独立、无原子竞争，一趟写回；映射与已实现的 `slot_pack_scale` 完全一致 | −20…−25 ms |
| **`fill_` 审计** | 75.4 ms（27× `[E,10,64]` 26.0 + 24× `[E,640]` 23.1 …） | 若这些是 autograd 为未用分支物化的零梯度，`set_materialize_grads(False)` / 改 `zeros` 为 `empty`+写就能去掉 | 待定位站点，可能 −30…−40 ms |
| **C5 readout** | 55.4 ms（`[1,152,48]×[1,48,576000]` 一组） | `float4` + 融合 scale/add epilogue，K=48 空转 | −15…−20 ms |

顺序建议：**C1b 先做**（有现成映射、自写 kernel、与 ridge 无关），
然后用一次 trace 把 `fill_` 的来源定位（看 launch 前后的 op 对，不需要 stack），
再决定 C5。

## 复现

```bash
python3 tests/deepmd/profile/chain_runs.py ../dpa4_inference_trace.json --top 12
```

回归测试（`tests/python/test_chain_runs.py`）在 trace 存在时核对上面三个数：
runs 合计 100.7 ms、len≥3 为 16.2 ms、`fill_` 配对数 0。
