# 已实现的算子融合：表达式与 DeepMD 原位代码

三个算子，共同点是把"多次 launch + 行原子散射"压成**一次 CSR 段遍历**。其中两个已经绑进 DeepMD 的分派接缝，
第三个的上游接缝（`edge_force_virial`）也已经存在，只差把 CUDA 实现换成我们的。

| 算子 | 表达式 | DeepMD 原位 | 原来 | 现在 | 状态 |
| --- | --- | --- | --- | --- | --- |
| `segment_sum` | `out[n,c] = s[n]·Σ_{dst(e)=n} w[e]·src[e,c]` | `sezm_nn/so2.py:1860-1873` | 加权 + 零初始化 + 原子散射 + 归一共 4 次 launch | 1 次 | ✅ 已接（`envelope` 接缝） |
| `segment_softmax` | `α_e = w_e/D`，见下 | `sezm_nn/attention.py:100-131` | max + exp + 2 次散射 + 除共 6 次 launch | 1 次 | ✅ 已接（`softmax` 接缝） |
| `force_assembly` | `F_k = Σ_{dst=k}g − Σ_{src=k}g`，`W_k = Σ_{src=k}(−g⊗v)` | `model/transform_output.py:300-345` | 2 次原子散射 + `(E,9)` 物化 | 2 次只读遍历 | ⬜ 接缝已存在，待接 |

profile 数据、候选排序与收益预算见 [`task2-hip-fusion-plan.md`](task2-hip-fusion-plan.md)；本文只讲**公式与代码对照**。

---

## 1. `segment_sum`

### 表达式

```
out[n, c] = s[n] · Σ_{dst(e)=n} w[e] · src[e, c]

w: 逐边权重（envelope，可含 source gate）
s: 逐节点后乘（1/√deg）
```

### DeepMD 原位：`deepmd/pt/model/descriptor/sezm_nn/so2.py:1860-1873`

```python
x_message, _ = self.edge_message(x, edge_cache, radial_feat)      # (E, D, C)
edge_weight = edge_cache.edge_env
edge_src_gate = edge_cache.edge_src_gate
if edge_src_gate is not None:
    edge_weight = edge_weight * edge_src_gate.to(dtype=edge_weight.dtype)
x_message = x_message * edge_weight.unsqueeze(-1)                 # ① 逐边加权
out = x.new_zeros(x.shape, dtype=self.compute_dtype)              # ② 零初始化
out.index_add_(0, edge_cache.dst, x_message.to(...))              # ③ 行原子散射
out.mul_(edge_cache.inv_sqrt_deg.to(...))                         # ④ 度归一
```

①②③④ 是 4 个 kernel；③ 在 60 边/原子时只有 ~177 GB/s（行原子串行）。

### 融合后

```python
offsets, order = csr_view(edge_cache.dst, n_nodes)          # DeepMD 已按步缓存
out = segment_sum(message.reshape(E, -1), dst, offsets,
                  edge_scale=env * gate, node_scale=inv_sqrt_deg, order=order)
```

块内按 `slot ∈ [offsets[n], offsets[n+1])` 走一遍，边号取 `order[slot]`；累加器在寄存器里，
**0 原子**，权重与归一分别作为 `edge_scale`/`node_scale` 进核 —— ①④ 直接消失，② 变成"直接写"。

前置条件：fp32、CUDA、`dst` int64；不满足则回退原方法。

---

## 2. `segment_softmax`

### 表达式（逐 destination、逐通道）

```
l_e  = logit_e + 2·ln env_e        (env_e > 0，否则该边不参与)
m    = max(l_0, max_e l_e)
w_e  = exp(l_e − m),  w_0 = exp(l_0 − m)
D    = w_0 + Σ_e w_e
α_e  = w_e / D,  α_0 = w_0 / D

一阶反向（用已存的 α，不重读 logits）：
S    = Σ_e α_e ᾱ_e + α_0 ᾱ_0
∂l_e = α_e(ᾱ_e − S),  ∂l_0 = α_0(ᾱ_0 − S),  ∂env_e = (Σ_c ∂l_e)·2/env_e
```

### DeepMD 原位：`deepmd/pt/model/descriptor/sezm_nn/attention.py:100-131`

```python
dst_index = dst.reshape(n_edge, 1).expand(n_edge, n_channel)

group_max = null_logit.expand(n_nodes, n_channel).clone()              # ① 初值
group_max = torch.scatter_reduce(group_max, 0, dst_index,
                effective_logits, reduce="amax", include_self=True)    # ② 段最大
edge_max  = group_max.index_select(0, dst)                             # ③ 广播
edge_exp  = torch.exp(effective_logits - edge_max)                     # ④ exp
denom_sum = torch.zeros(n_nodes, n_channel, dtype=compute_dtype, device=logits.device)  # ⑤
denom_sum = torch.scatter_add(denom_sum, 0, dst_index, edge_exp)       # ⑥ 段求和
denominator = denom_sum + torch.exp(null_logit - group_max)            # ⑦ null mass
alpha = edge_exp / denominator.index_select(0, dst)                    # ⑧ 归一
```

### 融合后

```python
null_logit = torch.log(F.softplus(z_bias_raw) + eps).reshape(1, C).expand(N, C)
alpha = segment_softmax(logits.reshape(E, C), env, offsets,
                        null_logit=null_logit.contiguous(), order=order)
```

一个 block 管一个 destination：`m`、`D` 留在寄存器，`(E,C)` 只走几遍；反向只读 `α`。
`env <= 0` 的边在核内直接跳过（等价于参考实现的 `-inf` 掩码）。

前置条件：`src_weight is None`（SFPG 桥接模式需要"线性 source ratio"，log 空间里没有等价位形，
因此该模式一定回退）、fp32、`z_bias_raw.numel() == F·H`。

**算子级实测**（181838 边 / 3000 原子）：4.42 → 0.49 ms（C=64，9.09×）、9.52 → 1.91 ms（C=256，5.00×）。

**端到端实测**（2026-11-19，DPA4-Air-OMol25 100M，3000 原子 / 181838 边，`n_atten_head=1`）：

| 指标 | 基线 | 打上 `--fused on` | 说明 |
| --- | --- | --- | --- |
| 接缝命中 | — | `softmax=18` / `fallback=0` | 3 block × 6 次调用（5 warmup + 1 profile），链路通 |
| 设备 kernel 数 | 2614 | 2464（**−150**） | 参考链的前向 8 个 + 反向约 40 个/次，全被替换 |
| 设备总时间 | 964.4 ms | 963.8 ms（**1.001×**） | 几乎没有收益 |

结论：**这条接缝在这个 checkpoint 上净值约 0**。它的宽度是 `attn_n_focus × 1`（`n_atten_head=1`），
被替换掉的都是 `[3000, C]` 量级的小张量，单次成本由 launch 开销主导（同一份 trace 里 `fill_` 只降了 0.14 ms）。
之前按 trace 里 `fill_[3000,256]` 归因到 softmax 是错的 —— 那些属于别的阶段。

真正的收益在 E-major（边流）的散射上：`force_assembly` 接 `edge_force_virial`，
以及 dense 路径里 `out_heads.index_add_`（见 §5）。

---

## 3. `force_assembly`

### 表达式

```
F_k = Σ_{dst(e)=k} g_e − Σ_{src(e)=k} g_e
W_k = Σ_{src(e)=k} (−g_e ⊗ edge_vec_e)      # 9 个分量，行主序
```

### DeepMD 原位：`deepmd/pt/model/model/transform_output.py:300-345`

```python
dst_order = torch.argsort(dst_ext)
src_order = torch.argsort(src_ext)
boundaries = torch.arange(n_ext + 1, device=g.device, dtype=dst_ext.dtype)
dst_row_ptr = torch.searchsorted(dst_ext.index_select(0, dst_order), boundaries)
src_row_ptr = torch.searchsorted(src_ext.index_select(0, src_order), boundaries)
if use_fused_cuda:
    force_flat, av_flat, frame_virial, _ = fused_edge_force_virial(
        g.contiguous(), edge_vec.detach().contiguous(), edge_scatter_index,
        edge_mask, dst_order, dst_row_ptr, src_order, src_row_ptr,
        n_node_per_frame, edge_vec.new_empty(0), n_ext, True)
```

注释里已经写明动机：*"two CSR segment reductions instead of four `index_add` scatters (which serialize on
the colliding edges of each atom) and a materialized `(E, 9)` outer product"*。

注意：这段是 **autograd 之外**的后处理，不在梯度图里；而且上游已经提供了 `fused_edge_force_virial`
这个接缝，只是绑的是 `pt_expt/kernels/edge_force_virial.py`（CUDA），DCU 上 `cuda_infer_level()` 不生效，
所以 trace 里仍是四次 `index_add_`。

### 融合后

```python
force, virial = force_assembly(g, edge_vec, dst_offsets, src_offsets, src_order)
```

一个 block 带 5 个原子（60 lane 填满一个 wave64），lane 0–2 算力、3–11 算维里，`double` 累加，
零原子、无 `(E,9)` 中间量、累加顺序确定。签名与上游 `fused_edge_force_virial` 的
`(g, edge_vec, edge_scatter_index, edge_mask, dst_order, row_ptr, src_order, row_ptr, …)`
是同构的，接线时只需在同一处做一次替换。

**实测**（E=181838）：`torch` 链 1.006 → 0.182 ms，**5.53×**。

---

## 4. 接线现状

| 算子 | DeepMD 名字 | 接法 | 开关 |
| --- | --- | --- | --- |
| `segment_softmax` | `sezm_nn.attention.segment_envelope_gated_softmax` | 模块级替换（`so2` 里的同名 import 一起换） | `fused=("softmax",)` |
| `segment_sum` | `SO2Convolution.forward_envelope` | 实例方法替换 | `fused=("envelope",)` |
| `force_assembly` | `pt_expt.kernels.edge_force_virial.edge_force_virial` | 待接（签名已对齐） | — |

细节见 [`deepmd-fused-dispatch.md`](deepmd-fused-dispatch.md)（接缝、前置条件、回退计数）。

## 5. 还没做的两项（设计，未实现）

| 候选 | 池子（3000 原子实测） | 为什么没做 |
| --- | --- | --- |
| C4 边流逐元素链融合（`add_/mul/copy_/fill_`） | 587.1 ms | 需要"项表"式逐元素内核；收益来自 pass 数 k→1，但模板要先从 n-gram 里挑 |
| C5 SO(2) 消息链（rotate→radial→GEMM→rotate-back→散射） | ~130 ms | 需要 HIP 版 block-diagonal 旋转，且与 DMK 的 GEMM epilogue 耦合（`_flash_atten_fn` 接缝） |

## 6. 验证

```bash
# 主机：参考实现 + 闭式反向 + 校验路径（不需要 DCU）
PYTHONPATH=python python3 -m unittest discover -s tests/python -v

# DCU：算子精度/梯度
PYTHONPATH=python python3 -m unittest discover -s tests/csrc/accuracy -p 'test_segment_ops*' -v

# DCU：与 DeepMD 自身两个函数的 parity（不需要 checkpoint）
DP_RUN_FUSED_DEEPMD_PARITY=1 \
  PYTHONPATH=python python3 -m unittest discover -s tests/csrc/accuracy -p 'test_fused_deepmd_parity.py' -v

# DCU：A/B 延迟（真实尺寸）
DPA4_RUN_SEGMENT_LATENCY=1 DPA4_SEGMENT_LATENCY_JSON=/tmp/seg.json \
  PYTHONPATH=python python3 -m unittest discover -s tests/csrc/latency -p 'test_segment_ops*' -v
```
