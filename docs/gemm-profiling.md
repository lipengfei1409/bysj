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

## 2. 寄存器压力与占用

`hipprof --codeobj-analyze <so>` 是**交互式**的：它列出 ELF 文件让你选序号，再选 kernel。想直接要
一张表就用仓库里的脚本——同一份数字（amdgpu metadata 里的 `NumVgprs` / `NumSgprs` /
`GroupSegmentFixedSize`）：

```bash
python3 tools/kernel_resources.py                 # 自动找 python/kernels/*.out
python3 tools/kernel_resources.py <x.out> --threads 256 --lds 24576
```

输出每行一个 kernel：`vgpr / sgpr / lds / blocks-per-CU`，occupancy 直接算好：

```
blocks/CU = min( 65536 / (threads × vgpr) , 65536 / (lds + 页) , wave 上限 )
```

如果 `*.out` 还不存在，先让 hipprof 跑一次 `--codeobj-analyze`（它会写出
`*.so-hipv4-amdgcn-amd-amdhsa--gfx928.out`），或直接 `roc-obj-extract`。

配合 `DPA4_DEBUG_GEMM=1` 打印的 `static_lds` 与真实 `regs` 互相对照，两个数一致就说明算得对。

## 2.5 已经测到的基线（2026-11-28，gfx928）

```
hipprof --stats:  dense_gemm_kernel   10 calls   avg 2.691 ms
bench_gemm dense --shape 2048x2048x2048 --iterations 5:  2.687 ms   6.39 TFLOP/s
```

反推每 CU 的预算（2048³、120 CU、~1.7 GHz）：

| 量 | 值 |
| --- | --- |
| 总 MMA 数（m16n16k8） | 1024 blocks × 256 k-steps × 16 = **4.19 M** |
| 每 CU 的 MMA 数 | 34.9 k |
| 每 k-step 的墙钟 | ≈ 2090 cycles |
| **每 MMA 的墙钟** | **≈ 131 cycles** ← 而 mma 本身只有 ~32 |

131 cycles 里 mma 只占 32：剩下的是 **LDS 取片（20 次 `ld_matrix`/k-step）、两次屏障、以及
未被重叠的全局装载延迟**。这就是为什么下一步要动 occupancy 与 k-stage，而不是继续调 tile。

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
