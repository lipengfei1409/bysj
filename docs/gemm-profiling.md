# 用 hipprof 定位 dense GEMM

这台 DCU 的 profiler 是 DTK 自带的 **`hipprof`**（`/opt/dtk-26.04/hip/bin/hipprof`），
不是 `rocprofv3`。它有 PMC 选项，**但不接受 `--` 分隔符**——命令直接跟参数即可，
`--` 会被判为 `Invalid arg`。

```bash
cd ~/private_data/lipengfei/bysj
source /opt/dtk/env.sh
export PATH=/opt/dtk/bin:$PATH
```

## 1. 每个 kernel 的时长与调用次数

```bash
hipprof --stats -o /tmp/dense_stats \
  env PYTHONPATH=python python3 -m dmk.scripts.bench_gemm dense \
      --shape 2048x2048x2048 --iterations 5
ls -l /tmp/dense_stats*
```

先确认 `dense_gemm_kernel` 的平均时长与 `bench_gemm` 打印的一致，并**看有没有别的 kernel**
混进来（有就是白送开销）。

## 2. 寄存器压力与占用（不用跑，直接看 ELF）

```bash
ls python/kernels/_C*.so
hipprof --codeobj-analyze python/kernels/_C*.so
```

这一项给的是每个 kernel 的 VGPR/SGPR 用量——**决定 occupancy 的另一半**。
配合 `DPA4_DEBUG_GEMM=1` 打印的 `static_lds` 就能算出每 CU 能放几个 block：

```
blocks/CU = min( 65536 / static_lds , 65536 / (threads × vgpr) )
```

## 3. PMC（计数器）

```bash
hipprof --pmc --pmc-type 3 --kernel-name dense_gemm_kernel -o /tmp/dense_pmc \
  env PYTHONPATH=python python3 -m dmk.scripts.bench_gemm dense \
      --shape 2048x2048x2048 --iterations 5
ls -l /tmp/dense_pmc*
```

`--pmc-type 3` 是 CSV 表。若本版需要显式给指标列表，先看
`ls -R /opt/dtk/hipprof_utils | head -40` 里有没有 pmc 配置文件，或试 `--pmc-read` / `--pmc-write`。

要看的指标（名字按本版 `--pmc-type 0` 的输出确认）：

| 指标 | 判读 |
| --- | --- |
| LDS bank conflict / LDS 指令数 | 接近 1 → 冲突主导（我们现在用 padding 缓解，见 `csrc/gemm.cpp`） |
| MFMA 指令数 × 4096 FLOP ÷ 时长 | 远低于实测 TFLOP/s → 矩阵单元在等数据 |
| 平均 wave 数（occupancy） | 只有 4（一个 block）→ 装载与计算串行 |
| TCC hit / miss | hit 低 → 每个输出 tile 都在重读 A/B，缺 L2 分块 |

## 4. 没有计数器时的替代：受控变量实验

见 [`gemm-knobs.md`](gemm-knobs.md)：`DPA4_DENSE_PAD` / `_KSTAGE` / `_CCHUNK` 各改一个决策，
每次只重编一个 TU，量同一形状。这一条不依赖 profiler，且能直接指向要改的那一行代码。
