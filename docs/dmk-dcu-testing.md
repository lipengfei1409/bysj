# 在 DCU 上测持久化 GEMM：完整命令

从零到"精度 + 性能"一条链，每一步都能独立失败，出错就知道停在哪一层。
仓库根目录以 `$REPO` 表示（DCU 上是 `~/private_data/lipengfei/bysj`）。

## 0. 环境

```bash
export REPO=~/private_data/lipengfei/bysj
cd "$REPO"
source /opt/dtk/env.sh            # DTK 26.04：hipcc / rocBLAS / rocprofv3

which hipcc rocprofv3             # 两个都要有
python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
#   期望：2.x  <hip 版本>  True
```

可选：只让进程看到一张卡，先排除多卡干扰。

```bash
export HIP_VISIBLE_DEVICES=0
```

## 1. 编译扩展（`kernels._C`，含 `csrc/dmk/mk_harness.cpp`）

```bash
export PYTORCH_ROCM_ARCH=gfx928        # 必须是实际卡的架构，见下
MAX_JOBS=8 python3 setup.py build_ext --inplace
```

**`PYTORCH_ROCM_ARCH` 是必须的**：DTK 的 `du_mma.h` 把每个 fragment 偏特化和每个
load/mma/store 函数都按架构 gate（只允许 `gfx926 / gfx928 / gfx936 / gfx938`），而 torch 默认会
把 gfx906 也算进去。表现是两层：

* gfx906 的 device pass 撞上 `#error "Only support ..."`；
* 于是报错里还会漂出 `implicit instantiation of undefined template DUFragment<...>` —— **那不是
  `ops/dumma.h` 写错了**，是这个架构根本没有偏特化。

先用 `rocminfo | grep gfx` 查本机架构。`setup.py` 现在会在 `PYTORCH_ROCM_ARCH` 含不支持的架构时
直接退出并给出这条提示；`ops/dumma.h` 也会在 device pass 里用 `DPA4_DUMMA_AVAILABLE` 把
"不支持的架构"变成一条明确的 `#error`。

**如果本卡就是 gfx906**，DUMMA 在这张卡上不存在（这是硬件事实），只能走标量分支。

如果本机 DTK 的 `du_mma.h` 不提供 fragment API（`DUFragment` 只有声明、没有
`du_load_matrix_sync` / `du_mma_sync` / `du_store_matrix_sync`），先用标量分支把调度侧跑通：

```bash
DPA4_GEMM_SCALAR=1 MAX_JOBS=8 python3 setup.py build_ext --inplace
```

标量分支与 DUMMA 分支共享同一套页/波/切分逻辑，只是内层换成 FFMA；它用来验证"调度对不对"，
不用来比性能。

`setup.py` 会检查 `torch.version.hip`，在非 HIP 的 torch 上直接报错退出——那时是环境没 source。

**这一步要过的门**：`ops/gemm.h` 的 DUMMA 模板（`DUFragment` 的 `(M,N,K)`、`kStep=4` 与
`mma_supported<float,4>`）。若报 `unsupported DUMMA (dtype, K) pair`，把 `GemmOp::kStep` 改成 8 并
把内层 `step += kStep` 一起改。

## 2. 主机侧回归（不需要卡，10 秒）

```bash
python3 tests/run_python_tests.py
#   期望：Ran 289 tests ... OK (skipped=2)
```

## 3. 单 tile 冒烟（**最先查数值**，30 秒）

```bash
PYTHONPATH=python python3 - <<'PY'
import torch
from dmk import dispatch, emitters, mk

def run(m, n, k, cus):
    g = mk.GemmGlobals.from_shapes(m, n, k, seed=0)
    g.attach(emitters.GemmEmitter(m, n, k).queues(cus))
    dispatch.launch_gemm(g.instructions, g.a, g.b, g.c)
    torch.cuda.synchronize()
    return (g.c - g.a @ g.b).abs().max().item()

# 16³：1 个 tile、1 个 k-step、只用到 wave 0；32×32：2 个 tile，用到 wave 0/1
for shape in ((16, 16, 16), (32, 32, 32), (64, 48, 32)):
    print(shape, "max err", run(*shape, cus=1))
# 16×256：一行 16 个 tile → 4 条 instruction，四个 wave 全用上
print((16, 256, 32), "max err", run(16, 256, 32, cus=1))
PY
```

`16³` 就错 → 只可能是 `ld_matrix` 的 `ld`、`store_matrix` 的 `mem_row_major` 或两个累加器交替；
`16×256` 才错 → 是 wave 划分（`tile_column` / `pid(2+wave)`）的问题。

## 4. 完整精度（11 形状 × {1,4,32} CU）

```bash
PYTHONPATH=python python3 -m unittest tests.csrc.accuracy.test_persistent_gemm -v
```

覆盖 16³ / 32³ / 64×48×32 / 256×16×64 / 32×32×128 / 128³ / 512³ / 1024³ / 2048³ /
2048×512×2048 / 512×2048×512；另含"与 CU 数无关"、"重复 launch 不累加"、"CU 多于 tile"、
"非 16 倍数被拒"。

## 5. 与 rocBLAS 比时间（大形状）

```bash
DPA4_RUN_PERSISTENT_GEMM_LATENCY=1 \
DPA4_PERSISTENT_GEMM_JSON=/tmp/gemm_latency.json \
DPA4_PERSISTENT_GEMM_ITERATIONS=20 \
DPA4_PERSISTENT_GEMM_REPEATS=5 \
DPA4_PERSISTENT_GEMM_WARMUP=5 \
PYTHONPATH=python python3 -m unittest tests.csrc.latency.test_persistent_gemm_latency -v
```

形状：512³ / 1024³ / 2048³ / 4096³ + 2048×8192×2048 + 8192×2048×2048 + 4096×4096×1024 +
1024×1024×4096，CU ∈ {4, 32, 120}。

每行形如：

```
  cus  32: persistent  12.345 ms (  9.87 TFLOP/s) | rocBLAS   8.765 ms ( 13.90 TFLOP/s) | 0.710x
```

`ratio = rocblas_ms / persistent_ms`，>1 表示我们更快。

跳过 rocBLAS（只看自己）：`DPA4_ROCBLAS=0`。
rocBLAS 包装编译/链接失败时不会让测试红，会打印原因；排查用 `DPA4_ROCBLAS_CPP_VERBOSE=1`。

## 6. Profile（可选，一次约两分钟）

```bash
# 6.1 一次 launch 的时长与资源
PYTHONPATH=python rocprofv3 --kernel-trace -f csv -o /tmp/gemm_k -- \
  env SHAPE=2048 CUS=32 python /tmp/bench.py

# 6.2 决定"下一步改哪"的计数器
rocprofv3 --list-pmc | head -30        # 先看本 DTK 的计数器名
PYTHONPATH=python rocprofv3 --pmc \
  GRBM_GUI_ACTIVE,SQ_INSTS_VALU,SQ_INSTS_LDS,SQ_LDS_BANK_CONFLICT,SQ_WAVES \
  -f csv -o /tmp/gemm_pmc -- env SHAPE=2048 CUS=32 python /tmp/bench.py
```

`/tmp/bench.py` 见 `docs/dmk-profiling.md`。判读表也在那里。

## 7. 一键（1→5）

```bash
set -e
export REPO=~/private_data/lipengfei/bysj
cd "$REPO" && source /opt/dtk/env.sh
MAX_JOBS=8 python3 setup.py build_ext --inplace
python3 tests/run_python_tests.py
PYTHONPATH=python python3 -m unittest tests.csrc.accuracy.test_persistent_gemm -v
DPA4_RUN_PERSISTENT_GEMM_LATENCY=1 PYTHONPATH=python \
  python3 -m unittest tests.csrc.latency.test_persistent_gemm_latency -v
```

## 失败对照

| 症状 | 位置 | 原因 |
| --- | --- | --- |
| `native build requires a ROCm/DTK PyTorch` | 1 | 没 `source /opt/dtk/env.sh` |
| `unsupported DUMMA (dtype, K) pair` | 1 | `kStep=4` 与本 DTK 不符 → 改 8 |
| 16³ 全错 | 3 | `ld_matrix` 的 `ld` / `store_matrix` 的 layout / 累加器交替 |
| 全 0 | 3 | `store()` 没被执行（storer 波序号不是 0） |
| 16³ 对、16×256 错 | 3 | wave 划分：`tile_column` 越界裁剪或 `pid(2+wave)` |
| 只有大 K 偏差 | 4 | fp32 累加顺序不同，`rtol=1e-4` 内属正常 |
| 精度全过、比 rocBLAS 慢 5–10× | 5 | 预期之内：单波 A 复制 4 份、无 L2 分块、无宏 tile |
