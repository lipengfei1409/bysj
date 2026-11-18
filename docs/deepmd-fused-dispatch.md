# 把融合算子接进 DeepMD：两个接缝与补丁位置

本文回答"这些 HIP 算子怎么在 DeepMD 框架里被调用"。结论：**不修改 DeepMD 源码，走它自己已有的两个插件点**，
补丁全部在 `myproject/python/model/` 下（`fused_deepmd.py` + `dpa4.py` 的一个 `fused=` 开关）。

```python
from model.dpa4 import DPA4Model

model = DPA4Model(model_file, fused=True)        # 安装两个接缝
print(model.fused_ops)                            # {'softmax_hits': 3, ...}
energy, force, virial = model.eval(coords, cells, types)
model.close()                                     # 自动 uninstall
```

---

## 1. 两个接缝

| # | DeepMD 侧的接缝 | 调用点 | 我们替换成 | 前置条件（否则回退） |
| --- | --- | --- | --- | --- |
| S1 | `segment_envelope_gated_softmax(logits, edge_env, dst, n_nodes, z_bias_raw, eps, src_weight=None)` | `sezm_nn/so2.py::SO2Convolution.attention_weights` 的 else 分支（`so2.py:2242`） | `kernels.segment.segment_softmax` | `src_weight is None`（SFPG 关闭）、fp32、CUDA、`logits` 3 维，`z_bias_raw.numel() == F*H` |
| S2 | `SO2Convolution.forward_envelope(self, x, edge_cache, radial_feat)` | `block.py` 的 SO(2) unit（`n_atten_head == 0` 的无注意力路径） | `kernels.segment.segment_sum` | 推理模式、fp32、`dst` int64、`edge_env`/`inv_sqrt_deg` 形状匹配 |

两个接缝都是**官方结构**，不是硬改：S1 是 DeepMD 自己在没有 Triton 时调用的参考实现（`attention.py`），
S2 是 `SO2Convolution` 的公开方法。`install()` 只是把这两个名字绑成我们的包装器，并保留原实现用于回退。

### S1 的对应关系

DeepMD 参考实现（`attention.py`）：

```
log_weight   = 2·ln(env)                        # env ≤ 0 的边置 -inf
effective    = logits + log_weight              # (E, F·H)
group_max    = scatter_reduce(amax) over dst    # 以 null_logit 初始化
denom        = scatter_add(exp(effective - m)) + exp(null_logit - m)
alpha        = exp(effective - m) / denom[dst]
```

我们的算子（`csrc/segment_softmax.cpp`）：同样的式子，但把 `m` 和 `D` 留在寄存器里、按 CSR 段走一遍，
并且**把 `2·ln(env)` 折进核内**（所以传 `env` 而不是 `log_weight`）。`null_logit` 由包装器算好：

```python
null_logit = log(softplus(z_bias_raw) + eps)          # (F·H,)
```

### S2 的对应关系

DeepMD 参考实现（`so2.py::forward_envelope`）：

```
message     = self.edge_message(x, edge_cache, radial_feat)   # (E, D, C)
edge_weight = edge_env * edge_src_gate                        # (E,)
out         = zeros(N, D, C)
out.index_add_(0, dst, message * edge_weight)                 # 原子散射
out.mul_(inv_sqrt_deg)                                        # (N,1,1)
```

我们的算子：`segment_sum(message[E, D*C], dst, offsets, edge_scale=edge_weight, node_scale=inv_sqrt_deg, order=order)`。
**一次遍历同时完成零初始化、散射与度归一**，没有原子、没有中间张量；`message * edge_weight` 这一步也消失了
（权重作为 `edge_scale` 进核）。

---

## 2. 为什么需要 `order`：DeepMD 的 `dst` 不是分段有序的

DeepMD 的边表按 **dst 分组但不保证位置有序**，所以 `index_add_` 只能用原子；`trace` 里这一族只有 177 GB/s。
DeepMD 自己也为我们准备好了 CSR 视图：`edge_cache.py::cached_edge_csr(edge_cache, "dst", n_node)`
返回 `(order, row_ptr)`，其中 `order` 是稳定 argsort 的置换、`row_ptr` 是 `(N+1,)` 前缀和，并且**按步缓存**
（它的 Triton/CUDA 算子共用同一份）。

因此这批 kernel 新增了可选的 `order` 参数：段内遍历 `slot ∈ [row_ptr[n], row_ptr[n+1])`，
边号取 `order[slot]`，**输出仍写在自然边序上**（`alpha[e]`、`grad_src[e]`），不需要再散回去。
为此改动：

- `csrc/segment_sum.cpp`：前向（标量与 `float4` 两条路径）与 `grad_node_scale` 接受 `order`；`order == nullptr` 时退化为原来的位置序。
- `csrc/segment_softmax.cpp`：前向与一阶反向的段内遍历接受 `order`。
- `python/kernels/segment.py`：`segment_sum(..., order=)`、`segment_softmax(..., order=)`。
- 包装器 `fused_deepmd._CsrView`：S1 只拿到 `dst`（`attention_weights` 没有把 `edge_cache` 传进参考函数），
  所以按 `dst` 的 storage 做一个小 LRU（容量 4）；S2 有 `edge_cache`，但没有调用 `cached_edge_csr`
  （避免依赖内部实现）——两者用同一个 `_CsrView`，一次 argsort 约 1.5 ms/步，全模型共用。

> 为什么不直接把 `dst` 传给核做原子散射：那正是要修的东西（177 GB/s → 630 GB/s 的 3.5 倍）。

---

## 3. 回退策略：宁可慢，不能错

每个包装器在**每次调用**上检查前置条件，任一不满足就调用原实现，并计数：

| 计数器 | 含义 |
| --- | --- |
| `softmax_hits` / `softmax_fallbacks` | S1 命中 / 回退次数 |
| `envelope_hits` / `envelope_fallbacks` | S2 命中 / 回退次数 |
| `bound_softmax` / `bound_envelope` | 是否真的绑上了（没有 DCU 扩展时为 `false`） |
| `last_error` | 最近一次异常（被 `warnings` 报一次，不会中断推理） |

典型回退原因：`src_weight` 非空（SFPG 桥接模式，参考实现里有一个"线性 source ratio"，
不能简单塞进 log 空间）、非 fp32、非 CUDA、`dst` 不是 int64、形状不匹配。
`install()` 在没有可用扩展时是**纯 no-op**，`fused_ops` 会如实显示 `bound_* = false`。

---

## 4. 还没接的三处（以及为什么）

| 位置 | 状态 | 原因 / 需要的下一步 |
| --- | --- | --- |
| 力/维里装配 `deepmd/pt/model/model/transform_output.py::edge_energy_deriv` | 未接 | 里面是 `torch.autograd.grad(energy, edge_vec)`，散射发生在 autograd 图内部。要接 `force_assembly` 得把"逐边梯度 → 原子力"这一步挪出图（`create_graph=False` 的推理专用路径），属于下一阶段，且要重新验证力/维里的符号与 ghost 布局 |
| flash 聚合 `SO2Convolution._flash_atten_fn` | 未接 | 签名是 `(x_local, rotation, rotate_inv_rescale, attn_alpha, order, row_ptr, dst, lmax, heads)`，内部要做 block-diagonal rotate-back。这正是 C5，需要先有 HIP 版旋转，才能把 `segment_sum` 接进去 |
| dense 注意力路径 `so2.py:2124` 的 `out_heads.index_add_` | 未接 | 内联在 `forward_attention_dense` 里，没有接缝；要么方法级补丁（复制约 40 行），要么等上游加参数。本 checkpoint 有 attention 但 `_flash_atten_fn` 未绑定时走这条，所以这是 S1 之后剩下的下一块收益 |

另外注意：**不要碰 `_segment_softmax_fn` 那条 Triton 接缝**。它绑定的是 Triton 实现
（`pt_expt/kernels/triton/sezm/segment_softmax.py`），本机前端/后端 LLVM 代际不匹配，`tl.dot` 无法编译；
我们的 S1 走的是**参考实现那一支**，与 Triton 完全无关。

---

## 5. 验证

主机（不需要 DCU/DeepMD，用假模块 + 参考实现对照，10 个用例）：

```bash
cd myproject && PYTHONPATH=python python3 -m unittest tests.python.test_fused_deepmd -v
```

DCU（需要 DeepMD + 已编译扩展，算子级 parity，不需要 checkpoint）：

```bash
DP_RUN_FUSED_DEEPMD_PARITY=1 PYTHONPATH=python \
  python3 -m unittest tests.csrc.accuracy.test_fused_deepmd_parity -v
```

端到端（同一份 profile 协议，看 `fused_ops` 的命中数与总设备时间）：

```bash
DPA4_MODEL=<ckpt> DPA4_NPZ=<npz> PYTHONPATH=python python3 - <<'PY'
from model.dpa4 import DPA4Model
import numpy as np, torch
model = DPA4Model("<ckpt>", fused=True)
data = np.load("<npz>")
out = model.eval(data["coord"], data["box"], data["type"])
print(model.fused_ops)          # 期望 softmax_hits ≈ 3 × blocks，fallbacks = 0
model.close()
PY
```

期望：`softmax_hits ≈ 3 × n_blocks`（每 block 每步一次），`fallbacks = 0`。
单算子实测 5.0–9.1×（`docs/task2-hip-fusion-plan.md` §5），端到端收益取决于这一步占多少（见该文档 §7 的 C1–C3 折算）。

---

## 6. 文件

| 文件 | 作用 |
| --- | --- |
| `python/model/fused_deepmd.py` | 两个接缝的安装/卸载/计数、CSR 视图缓存、回退 |
| `python/model/dpa4.py` | `DPA4Model(..., fused=True/"softmax"/{"envelope": True})` 开关；`close()` 自动卸载 |
| `csrc/segment_sum.cpp`、`csrc/segment_softmax.cpp` | 新增可选 `order`（CSR 置换）参数 |
| `tests/python/test_fused_deepmd.py` | 主机：接缝命中/回退/卸载/幂等，与参考实现数值对照 |
| `tests/csrc/accuracy/test_fused_deepmd_parity.py` | DCU：与 DeepMD 自身两个函数的 parity |

---

## 附：与 Triton 版的关系

DeepMD 的 Triton 版把同两个位置做成 `_segment_softmax_fn`（CSR 分段 softmax）与 `_flash_atten_fn`
（旋转 + 加权 + 散射一体）。我们的 S1 是前者在 HIP 上的等价物（数学一致），S2 是无注意力路径的等价物；
S3（flash 聚合）要等 C5 的旋转核。也就是说：**Triton 版覆盖的三个位置，我们目前覆盖两个**，
剩下的那个恰好是最贵的一个（`_flash_atten_fn` 把 rotate-back 的中间量也省了）。
