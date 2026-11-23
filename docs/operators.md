# 融合算子：作用、调用点、时间占比、有效性

本文是**逐算子的唯一索引**：它是什么、在 DeepMD 的哪一行被调用、占多少设备时间、验证到哪一级。
优化项的完成度与收益汇总见 [`optimization-status.md`](optimization-status.md)，
时间占比的口径与全表见 [`op-time-share.md`](op-time-share.md)。

```python
from model.dpa4 import DPA4Model

model = DPA4Model(model_file, fused=True)     # 装全部接缝；也可 fused=("pack",)
energy, force, virial = model.eval(coords, cells, atom_types)
print(model.fused_ops)                        # 逐接缝的命中/回退计数
model.close()
```

## 0. 一页汇总

| 算子 | 作用（一句话） | 调用点 | trace 池 / 占比 | 有效性 | 端到端 |
| --- | --- | --- | ---: | --- | --- |
| `slot_pack_scale` | 按 gate 组展开并逐元素乘：`out[e,r] = gate[e,slot[r]]·x[e,r]` | `GatedActivation.forward`（`activation.py:213`，`fndc` 分支） | 53.4 ms / **4.4%** | L0 主机逐位对照；L2 未测 | 待测，估 −20…−25 ms |
| `segment_sum` | CSR 段归约：`out[n] = s[n]·Σ_{dst=n} w_e·src_e` | `SO2Convolution.forward_attention_dense` 的 `zeros+index_add_`（本模型走这条）；`forward_envelope`（无注意力路径） | 11.4 ms / 0.9% | L0 有；L2 **至今未测** | 待测，估 −10 ms |
| `segment_softmax` | 门控分段 softmax，`m`/`D` 留寄存器，`2·ln env` 折进核内 | `attention.segment_envelope_gated_softmax`（`so2.py:2242`） | ~0.15 ms / 0.01% | L2 **9.09×**；L4 **≈0** | ❌ 本 checkpoint 无效 |
| `force_assembly` | 力/维里装配：两次只读 CSR 遍历替代原子散射 + `(E,9)` 物化 | `transform_output.edge_force_assembly`（`transform_output.py:306-368`） | 1.7 ms / 0.1% | L2 **5.53×** | 池子太小，−1.5 ms |
| — 未接站点的算子 — | | | | | |
| `slot_reduce` | `Σ_r w[e,r]·src[e,r,c]` | **无**（真实站点权重依赖输出槽，见 §5） | 0 | 算子存在，无调用 | 0 |
| `gie_zonal_scatter` / `zonal_scatter` | 几何初嵌入的 zonal×radial gather → 散射 → 归一 | **无**（原 `pt_expt` patcher 已删） | 0 | 仅算子级测试 | 0 |
| `rmsnorm` | RMSNorm | **无**（本 checkpoint 走 DeepMD 自带） | 0 | 仅算子级测试 | 0 |

**合计**：四个接缝覆盖 **66 ms / 5.4%**，可实现收益 −32…−37 ms（约 1.03×）。

## 1. 有效性等级

| 级别 | 问的问题 | 判定物 | 需要 |
| --- | --- | --- | --- |
| **L0** 主机契约 | 数学与 API 对不对 | `tests/python`（109 项）、`reference/` 逐位对照 | 无（macOS 可跑） |
| **L1** 算子精度/梯度 | 与 float64 真值差多少 | `tests/csrc/accuracy/test_{segment,slot}_ops_accuracy.py` | DCU + 扩展 |
| **L2** 算子 A/B 延迟 | 同一表达式，比 ATen 链快多少 | `tests/csrc/latency/*`（真实尺寸 181838 边 / 3000 原子） | DCU + 扩展 |
| **L3** 接缝 parity | 与 DeepMD 自己的实现是否逐元素一致 | `test_fused_deepmd_parity.py` | DCU + 扩展 |
| **L4** 端到端 A/B | 真机推理快了多少 | profile CLI `--fused off/on` + `compare_fused_runs.py` | DCU + 模型 + 输入 |

一个算子只有走到 L4 才算"有效"。L2 只证明"这个核比这条链快"，L3 只证明"接对了"。

## 2. 逐算子

### 2.1 `slot_pack_scale` —— 53.4 ms（4.4%），最大的一个

**作用**：`GatedActivation` 里的 gate 展开。DeepMD 原位是三趟 E-major 遍历：

```python
gates = gating_scalars.index_select(dim=2, index=self.expand_index)   # [1,E,3,64] → [1,E,9,64]
out.narrow(axis, 1, D - 1).copy_(x.narrow(axis, 1, D - 1) * gates)    # mul + copy_
```

融合后一趟完成"按 gate 组展开 + 逐元素乘"：`slot_pack_scale(gating_scalars, expand_index, scale=x[:,1:])`。

**调用点**：`deepmd/pt/model/network/activation.py::GatedActivation.forward`，只接管 `fndc` 布局
（本模型就是它），另两条 layout、训练模式、非 fp32、`expand_index` 形状不符全部回退原方法。

**时间占比**（trace，各 9 次）：`index_select [1,E,3,64] idx[9]` 20.5 + `mul [1,E,9,64]` 17.3 +
`copy_ [1,E,9,64]` 15.6 = **53.4 ms**。融合后仍要读 `gates` 源（139.6 MB）、读 scale（419 MB）、
写 out（419 MB）≈ 978 MB → 9 次约 14 ms，再加末端 `narrow.copy_` 15.6 ms → 约 30 ms。

**有效性**：L0 用 `fndc` 假模块与 stock 方法**逐位对照**（3 个用例）+ 两条回退用例；
L2 的 `slot_pack_scale` 延迟测试至今没在真机跑过；L4 未测。**缺口：L2 和 L4。**

### 2.2 `segment_sum` —— 11.4 ms（0.9%）

**作用**：`out[n] = inv_sqrt_deg[n] · Σ_{dst(e)=n} (env_e·gate_e)·src_e`，把"零初始化 + 行原子散射 +
度归一"三趟合一次 CSR 段遍历，累加器在寄存器，**0 原子**（行原子只有 177 GB/s）。

**调用点**：两个接缝 —— `SO2Convolution.forward_attention_dense` 里的 `zeros + index_add_`（dense 注意力
路径，本模型走这条，`fused=("aggregate",)`）与 `forward_envelope`（无注意力路径，`fused=("envelope",)`）。
CSR 视图直接复用 DeepMD 的 `edge_cache.py::cached_edge_csr(edge_cache, "dst", n_node)`，
按步缓存，`order` 传进核内、输出仍写自然边序。

**时间占比**：`fill_ [3000,16,1,1,64]` 0.09 + `index_add_ [3000,16,1,1,64] ⬅ [E,16,1,1,64]` 11.15 +
`mul`（度归一）0.17 = **11.4 ms**，3 次（12 block 里只有 3 个走这条）。

**有效性**：L0 有；L1 有；**L2 从未测到**（首次 latency 运行被校验器 bug 中断后没重跑）；
L4 未测。收益上限 = 11.4 → 1.2 ms（0.745 GB @ 630 GB/s）→ **−10 ms**。

### 2.3 `segment_softmax` —— 0.15 ms（0.01%）

**作用**：逐 destination、逐通道的门控 softmax。参考实现是"初值 + 段 max + 广播 + exp + 段和 + null mass +
归一"共 8 个 aten 算子，融合后一个 block 管一个 destination，`m`/`D` 留寄存器，反向只读已存的 `α`。

**调用点**：`sezm_nn.attention.segment_envelope_gated_softmax`（模块级替换，`so2.py` 里的同名 import 一起换）。

**时间占比**：trace 里**整个前向 + 反向只有 3 次 `scatter_reduce [3000,1]` 共 0.12 ms**——
这条软最大作用在 `[3000, C]` 的 node-major 小张量上，因为 `n_atten_head=1`。
这就是 L4 实测 ≈0（964.4 → 963.8 ms，−150 launch）的原因，不是接线失败（`softmax_hits=18`,
`fallbacks=0`）。

**有效性**：L2 **9.09×**（C=64）/ **5.00×**（C=256）；L4 **≈0 → 本 checkpoint 判无效**；
它的价值要等 attention 宽度大的模型。

### 2.4 `force_assembly` —— 1.7 ms（0.1%）

**作用**：`F_k = Σ_{dst=k} g − Σ_{src=k} g`，`W_k = Σ_{src=k} (−g ⊗ v)`。参考实现是两次原子散射 +
一次 `(E,9)` 外积物化；融合后两次只读 CSR 遍历，一 block 带 5 个原子（60 lane 填满 wave64），
lane 0–2 算力、3–11 算维里，double 累加。

**调用点**：`transform_output.py:306-368`，替换 `pt_expt...triton.sezm.force_assembly.edge_force_assembly`
并把 `triton_infer_level()` 抬到 1 让分支生效（`fused=("force",)`）。DeepMD 在进分支前已经 `argsort`
出 `dst_order/src_order` 与两个 `row_ptr`，正好是核的入参。

**时间占比**：`index_add [3000,3]`（dst/src 各一次）1.18 + `index_add [3000,9]`（源端维里）0.40 +
`(E,3,1)×(E,1,3)` 外积 0.08 + `fill_` 0.03 = **1.7 ms**。之前记的"95 ms 池"是把 gate/槽位 gather
的反向误算进来了（见 [`op-time-share.md`](op-time-share.md) §5）。

**有效性**：L2 **5.53×**（1.006 → 0.182 ms）是真的，但池子只有 1.7 ms → **端到端最多 −1.5 ms**；
L4 已判定"不值得继续投入"，算子保留作 CSR 基建。

## 3. 接缝与回退契约

| # | DeepMD 接缝 | 我们替换成 | 前置条件（任一不满足即回退原实现） | 开关 |
| --- | --- | --- | --- | --- |
| S1 | `segment_envelope_gated_softmax` | `segment_softmax` | `src_weight is None`（SFPG 关闭）、fp32、CUDA、`logits` 3 维、`z_bias_raw.numel()==F·H` | `fused=("softmax",)` |
| S2 | `SO2Convolution.forward_envelope` | `segment_sum` | 推理、fp32、`dst` int64、`edge_env`/`inv_sqrt_deg` 形状匹配 | `fused=("envelope",)` |
| S3 | `forward_attention_dense` 的 `zeros+index_add_` | `segment_sum` | 块属性齐全、fp32、`dst` int64、`α` 形状 `(E,Fa,H)` | `fused=("aggregate",)` |
| S4 | `edge_force_assembly` + `triton_infer_level()` | `force_assembly` | 无 `create_graph`、fp32、两对 CSR 置换长度一致；cuTile 开启时跳过 | `fused=("force",)` |
| S5 | `GatedActivation.forward` | `slot_pack_scale` | `fndc` 布局、推理、fp32、`expand_index` 形状匹配 | `fused=("pack",)` |

公共契约：每次调用前检查前置条件；不满足走原实现并把异常记进 `FusedStats.last_error`，同时
`*_fallbacks` 加一。`uninstall()` 还原模块级名字并删除实例属性，所以 `fused=False` 与"没装"等价。

`fused_stats` 里要看的字段：`bound_*`（挂上了吗）、`*_hits`（真的被调用了吗）、
`*_fallbacks`（为什么没走）、`last_error`（回退原因）。**`hits=0` 而 `bound=true` 说明接缝条件没命中，
不是算子错。**

## 4. 快速验证阶梯

一条命令跑完（`verify_fused.py`，最后自动判三条端到端判据）：

```bash
cd ~/private_data/lipengfei/bysj/myproject
python3 tests/deepmd/profile/verify_fused.py --dry-run        # 只打印会跑什么，任何机器
python3 tests/deepmd/profile/verify_fused.py --stage host     # 主机契约，30 秒
python3 tests/deepmd/profile/verify_fused.py --stage all \
    --model-file "$MODEL" --input-npz "$NPZ"                  # DCU 上全跑
```

它按 `host → build → acc → lat → par → e2e` 顺序执行，e2e 段会读两份 summary 并打印
"接缝命中 / launch 数 / 设备时间"三条判据，任一不满足即以非零码退出。
下面每一步的原始命令与判据：

### L0 · 主机（30 秒，不需要 DCU）

```bash
cd ~/private_data/lipengfei/bysj/myproject
PYTHONPATH="$PWD/python" python3 -m unittest discover -s tests/python -v
```

判据：`109 passed, 1 skipped`。这一步覆盖参考实现、闭式反向、参数校验、`fndc` 逐位对照、
`uninstall` 还原、以及"DCU 测试不得引用已删 API"的守卫。

### L1 · 算子精度/梯度（5 分钟）

```bash
MAX_JOBS=4 python3 setup.py build_ext --inplace
python3 -m unittest discover -s tests/csrc/accuracy -p 'test_segment_ops*' -v
python3 -m unittest discover -s tests/csrc/accuracy -p 'test_slot_ops*' -v
```

判据：全部 `ok`；失败时先看是 fp32 累加顺序（正常，容差内）还是形状/索引（真 bug）。

### L2 · 单算子 A/B 延迟（10 分钟，唯一能直接换算收益的一步）

```bash
OUT=/root/private_data/lipengfei/test_result
DPA4_RUN_SEGMENT_LATENCY=1 DPA4_SEGMENT_LATENCY_JSON="$OUT/seg_latency.json" \
  python3 -m unittest discover -s tests/csrc/latency -p 'test_segment_ops*' -v
DPA4_RUN_SLOT_LATENCY=1 DPA4_SLOT_LATENCY_JSON="$OUT/slot_latency.json" \
  python3 -m unittest discover -s tests/csrc/latency -p 'test_slot_ops*' -v
```

判据：JSON 里每条记录的 `speedup`。**这是四个算子里唯一还缺的一格**（`segment_sum` 与
`slot_pack_scale` 都还没跑过）。注意 `slot_reduce` 的 A/B 测的是一个模型里不存在的表达式（§5），
它的数字不能用来估收益。

### L3 · 接缝 parity（5 分钟，不需要 checkpoint）

```bash
DP_RUN_FUSED_DEEPMD_PARITY=1 \
  python3 -m unittest discover -s tests/csrc/accuracy -p 'test_fused_deepmd_parity.py' -v
```

判据：与 DeepMD 自己的 `segment_envelope_gated_softmax` / `forward_envelope` 逐元素一致（fp32 容差）。

### L4 · 端到端 A/B（20 分钟，判"有没有用"的唯一标准）

```bash
OUT=/root/private_data/lipengfei/test_result
for one in off on; do
  flag=""; [ "$one" = on ] && flag="--fused on"
  python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
    --model-file "$MODEL" --input-npz "$NPZ" --fixed-input \
    --expected-dcu-name K500SM_AI $flag --output-dir "$OUT/fused_$one"
done
python3 tests/deepmd/profile/compare_fused_runs.py \
  --baseline "$OUT"/fused_off/dpa4_*/dpa4_inference_profile_summary.json \
  --patched  "$OUT"/fused_on/dpa4_*/dpa4_inference_profile_summary.json
```

判据三条，缺一不可：

1. **命中**：summary 的 `fused` 字段（CLI 也会打印 "Fused DeepMD dispatch (after inference)"）里 `pack_hits>0`、`aggregate_hits>0`、`force_hits>0`，
   所有 `*_fallbacks == 0`；
2. **launch 数下降**：`device_event_count` 明显变小（B2 单独就 −150）；
3. **设备时间下降**：`compare_fused_runs.py` 给出的 `device total` 变化。

**只有第 3 条为正才算有效。** 前两条通过而第 3 条不动，说明池子被高估（B2 就是这样）。

### 分阶段开，定位收益来自哪一项

```bash
for f in off softmax aggregate force pack; do
  args=""; [ "$f" != off ] && args="--fused $f"
  python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
    --model-file "$MODEL" --input-npz "$NPZ" --fixed-input \
    --expected-dcu-name K500SM_AI $args --output-dir "$OUT/one_$f"
done
```

## 5. 没有调用点的算子（别在它们身上花时间）

| 算子 | 为什么没有站点 | 处置 |
| --- | --- | --- |
| `slot_reduce` | 真实调用是 `Σ_r kernel[e,o,r]·x[e,r,c]·channel_basis[o,r,c]`，权重随**输出槽 `o`** 变化，不是按 `(e,r)` 的加权和 | 要么重写成 rank-K 收缩并入 C7，要么删掉 |
| `gie_zonal_scatter` / `zonal_scatter` | 原 `pt_expt`-only patcher 已随 `native_ops` 删除；legacy `pt` checkpoint 上本来也跑不起来 | 保留算子与精度测试，标"无 dispatcher" |
| `rmsnorm` | 本 checkpoint 的 norm 走 DeepMD 自带实现，没有接缝 | 同上 |

它们的延迟数字只能证明"这个核本身多快"，不能折算成模型收益。

## 6. 结论（2026-11-23）

- **已证明有效**：`segment_softmax` 算子级 9.09×（C=64）/ 5.00×（C=256）、`force_assembly` 5.53×；
  但两者的端到端收益分别 ≈0 与 −1.5 ms —— **算子快 ≠ 模型快**，这是本项目最重要的一条教训。
- **已接线、未测收益**：`slot_pack_scale`（53.4 ms 池）与 `segment_sum`（11.4 ms 池），
  预期合计 −30…−35 ms；最短路径就是上面的 L2 + L4。
- **已判无效**：`segment_softmax` 在本 checkpoint（`n_atten_head=1`）。
- **无站点**：`slot_reduce`、`gie`/`zonal_scatter`、`rmsnorm`。

## 7. 更新日志

| 日期 | 变更 |
| --- | --- |
| 2026-11-23 | 建表：把 `fused-operators.md`（表达式对照）与 `deepmd-fused-dispatch.md`（接缝/回退）合并为本文，并补上时间占比、有效性等级 L0–L4、四段验证阶梯；删除 6 份过期文档（见提交说明） |
