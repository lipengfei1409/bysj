# `SO2Linear` 分解

源码：`deepmd/pt/model/descriptor/sezm_nn/so2.py`（`class SO2Linear`，约 230–620 行）。
本文按 `DPA4-Air-OMol25` 的真实配置拆解：**`lmax=3`、`mmax=1`、`channels=64`、`n_focus=1`、
E=181 838 条边**。

## 1. 它在做什么

输入是**边**上的 SO(2) 约化特征，按 `m` 分块存放：

```
x: (F, E, D_m, Cin)        F = focus 流（batch 轴）, D_m = 约化系数维, Cin = 通道
```

输出同构、通道可换：`(F, E, D_m, Cout)`。它对每个 `m` 分量做**独立**的通道线性变换，
且 `|m|>0` 的 ±m 两分量之间用 2×2 耦合，从而在 SO(2) 旋转下等变。

## 2. 系数维 `D_m` 的来源

```python
group_sizes = [lmax + 1] + [2*(lmax - m + 1) for m in 1..mmax]
            = [4, 6]                       # m=0 组 4 个，|m|=1 组 6 个
reduced_dim = sum(group_sizes) = 10
```

对 `lmax=3`：m=0 有 `l=0,1,2,3` 四个系数（1+3+5+7 里 m=0 的那些），|m|=1 有 `l=1,2,3`
的正负两支共 6 个。展平后宽度 `10 × 64 = 640` —— 正是执行模型文档里的 `ROW = CF(3L+1) = 640`。

## 3. 权重结构：为什么是块对角

`_build_so2_weight()` 组装 `(D_m·Cin, F, D_m·Cout) = (640, 1, 640)`：

| 块 | 形状 | 参数 | 结构 |
| --- | --- | --- | --- |
| **m = 0** | 256 × 256 | `weight_m0` | 直接线性（标量分量不与 ±m 混合） |
| **\|m\| = 1** | 384 × 384 | `weight_m` | 2×2 耦合：`[[W_u, W_v], [-W_v, W_u]]` |

`|m|>0` 的耦合来自 SO(2)：旋转角 φ 时 `(u, v)` 变成 `(u cosφ - v sinφ, u sinφ + v cosφ)`，
所以权重必须长成 `[[W_u, -W_v],[W_v, W_u]]`（源码里按 `neg_in/pos_in × neg_out/pos_out`
四块填入，见 `_build_so2_weight` 的 `# Fill the 2x2 coupling` 段）。

**跨 `|m|` 的块恒为零**（不同 m 在 SO(2) 下不能混合）——这就是"块对角"的来源，
也是省掉近一半 FLOP 的依据。

## 4. 一次调用做几步

`forward()`（`so2.py:400`）：

```python
x_flat = x.reshape(F, E, 640)                    # ① 展平系数+通道
weight = self._build_so2_weight() or _cached_weight   # ② 组权重（推理时缓存）
out_flat = self._block_diagonal_matmul(x_flat, weight)  # ③ 逐块 bmm
out = out_flat.reshape(F, E, 10, 64)             # ④ 还原
out[:, :, 0, :] += bias0                          # ⑤ 只给 l=0 标量加偏置
```

`_block_diagonal_matmul`（`so2.py:556`）：

```python
weight = weight.permute(1, 0, 2)                 # (F, K, N) 视图，无拷贝
blocks = [torch.bmm(x_flat[:, :, i0:i1], weight[:, i0:i1, o0:o1])
          for i0, i1, o0, o1 in self._block_diag_slices]
return torch.cat(blocks, dim=-1)                 # 647 MB 的新缓冲区
```

另有两条等价路径：

| 路径 | 条件 | 说明 |
| --- | --- | --- |
| **dense einsum** | 仅 CPU | `einsum("fei,ifo->feo")`，块对角 → 稠密，慢一倍 |
| **block-diag bmm** | 非 CPU（默认） | 2 条 bmm，跳过结构零 |
| **Triton 融合** | `active_triton_level >= 1` | `pt_expt/kernels/triton/sezm/so2_block_gemm.py`：按块一次 launch，**直接写进输出、不 cat**，`BN=64` 避免 pow2 补齐 |

## 5. FLOP 与带宽账（E = 181 838）

| 块 | 宽度 | 每条 bmm | 每次调用 | 占比 |
| --- | --- | --- | --- | --- |
| m=0 | 256 | `2·E·256·256` = 23.8 GFLOP | | 31% |
| \|m\|=1 | 384 | `2·E·384·384` = 53.6 GFLOP | | 69% |
| **合计** | 640 | | **77.5 GFLOP** | |

- **一次前向 24 次调用** → 1859 GFLOP，与实测的 `1287 + 572` **完全吻合**。
- 若不走块对角（dense 640×640）：**149 GFLOP/次**，块对角省 **48%**。
- 实测这两个 bmm 落在 rocBLAS 的 `MT128x64x16` 上，共 **118.5 ms**（68.6 + 49.9），
  折合 **15.7 TFLOP/s**。

带宽（eager 路径，每次调用）：

```
bmm 写两个输出    466 MB
cat 读 + 写       931 MB
合计             1397 MB / 次   →  24 次 = 33.5 GB
```

这 33.5 GB 就是 Triton `so2_block_gemm` 要去掉的东西（"never concatenating a fresh output"）。

## 6. 优化含义

| # | 观察 | 手段 |
| --- | --- | --- |
| 1 | 两条 bmm 是独立 GEMM，`N=K=256/384`，`M=E` 极大 | 与我们的 `128×64` dense tile 完全同构；但要处理 **B 是 `permute(1,0,2)` 的 strided view** |
| 2 | `cat` 白花 931 MB/次 | 要么用 Triton 融合（已有），要么让 bmm 直接写进预留的 640 宽输出（切片写） |
| 3 | `520` 的 pow2 补齐：`N=192` 会补到 256 | Triton 用 `BN=64`；若自研 kernel，tile 宽度要能被 192 整除 |
| 4 | 每次调用只做 `K=256/384`（32–48 个 k8 步） | 流水能跑起来，但**启动开销/尾块**占比比 4096³ 大 |
| 5 | 24 次调用互相串行（层间有激活） | 无法跨层批量；可做的是**层内两条 bmm 与相邻逐元素融合**（C6/C7 的活） |

一句话：`SO2Linear` 本身已经把"跳过结构零"做掉了（省 48%），
剩下的空间在**不落 cat、不重排 stride、与前后逐元素算子合并**——这正是
`so2_block_gemm.py` 与 `gated_activation.py` 这类融合 kernel 在做的事。
