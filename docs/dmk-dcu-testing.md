# 在 DCU 上测持久化 GEMM：完整顺序

从"这张卡能不能用矩阵单元"开始，每一步只验一件事，失败时就知道停在哪一层。
仓库根目录以 `$REPO` 表示（DCU 上是 `~/private_data/lipengfei/bysj`）。

## 0. 先定架构——它决定这次能编哪条路径

```bash
export REPO=~/private_data/lipengfei/bysj
cd "$REPO"
source /opt/dtk/env.sh

rocminfo | grep -o "gfx[0-9]*" | sort -u
```

| 查到的架构 | 走哪条路 |
| --- | --- |
| `gfx926` / `gfx928` / `gfx936` / `gfx938` | **DUMMA**（矩阵单元），`export PYTORCH_ROCM_ARCH=<该架构>` |
| 其它（如 `gfx906`） | 这张卡**没有**矩阵单元：`export DPA4_GEMM_SCALAR=1`，性能预期改成标量上限（~18 TFLOP/s 理论峰值） |

`PYTORCH_ROCM_ARCH` 必须设：DTK 的 `du_mma.h` 把每个 fragment 偏特化和每个 load/mma/store 都按架构
gate，torch 默认还会编 gfx906，那会触发 `#error`，并在错误里漂出
`undefined template DUFragment<...>`（看着像代码 bug，其实是目标不支持）。`setup.py` 现在会在
架构列表不合法时直接退出并说明出两条路。

## 1. 编译

现在的 `setup.py` **只编框架**：

```
csrc/bindings_dmk.cpp      只有 dmk_gemm 一个入口
csrc/dmk/megakernel.cpp    唯一的 kernel（mk_all + launch）
```

`csrc/ops/` 是 header-only，所以这两个 TU 就是全部。其余手写 kernel（rmsnorm / gie / segment_* /
force_assembly / slot_ops / edge_combine / monomial / blockdiag / so2_conv）**暂时不编**——它们各自的
binding 在 `csrc/bindings.cpp` 里，等它们回来时把 `sources` 与 `bindings_dmk.cpp` 一起换回
`bindings.cpp` 即可。

```bash
MAX_JOBS=8 python3 setup.py build_ext --inplace
```

要过的门：
`ops/gemm.h` 的 fragment 声明、`dmk/launch.h` 的 LDS 预留、`ops/dumma.h` 的名字解析。
报 `unsupported DUMMA (dtype, K) pair` 或 `has no DUMMA` 就回第 0 步。

## 2. 主机侧回归（不需要卡，10 秒）

```bash
python3 tests/run_python_tests.py
#   期望：Ran 262 tests ... OK (skipped=2)
```

## 3. 冒烟：五个形状，各隔离一个机制（**最先跑的卡上测试**）

```bash
PYTHONPATH=python python3 -m unittest tests.csrc.accuracy.test_persistent_gemm_smoke -v
```

| 形状 | 隔离的机制 |
| --- | --- |
| 16³ | 单 tile、单 k-step、只有 wave 0：页 / stage / fragment / 尾声 |
| 16×32 | 一条指令两个 tile：wave 0 与 1 |
| 16×64 | 一条指令**四个** tile：四个 consumer wave + 四个 C 页 + storer 遍历 |
| 16×48 | 最后一个 run 只有三个 tile：**wave 3 必须完全不动**（越界裁剪） |
| 64×256 | 四行 tile，每行一个满 run |
| 128³ | 深 K：每波跨 8 个 k-tile 累加 |

失败时的分流：

| 失败在 | 说明 |
| --- | --- |
| `16³` | `ld_matrix` / `store_matrix` 的参数（`ld`、`mem_row_major`）或累加器交替 |
| `16³` 过、`16×32` 错 | 指令覆盖两个 tile 的 `n0` 计算（主机 `GemmEmitter.emit`） |
| 小形状过、`16×64` 错 | 四波并行：`pid(2 + wave)` 的 C 页分配或 `tile_column` |
| `16×48` 错 | 越界 wave 写了不该写的 C 页（`column < 0` 分支） |
| 全过、`128³` 错 | k 循环：每 k-tile 重新 stage 的位置（`wave * kSliceElements`） |
| **全 0** | `store()` 没执行——它是 storer 波，不能按 `wave == 0` 判断 |

## 4. 完整精度（11 形状 × {1,4,32} CU）

```bash
PYTHONPATH=python python3 -m unittest tests.csrc.accuracy.test_persistent_gemm -v
```

补上冒烟没覆盖的：512³ / 1024³ / 2048³、2048×512×2048、512×2048×512、CU 多于 tile、
重复 launch 不累加、非 16 倍数被拒。

## 5. 与 rocBLAS 比时间

```bash
DPA4_RUN_PERSISTENT_GEMM_LATENCY=1 \
DPA4_PERSISTENT_GEMM_JSON=/tmp/gemm_latency.json \
DPA4_PERSISTENT_GEMM_ITERATIONS=20 \
DPA4_PERSISTENT_GEMM_REPEATS=5 \
DPA4_PERSISTENT_GEMM_WARMUP=5 \
PYTHONPATH=python python3 -m unittest tests.csrc.latency.test_persistent_gemm_latency -v
```

形状 512³…4096³ 与三种矩形，CU ∈ {4, 32, 120}。每行：

```
cus  32: persistent  12.345 ms (  9.87 TFLOP/s) | rocBLAS   8.765 ms ( 13.90 TFLOP/s) | 0.710x
```

`ratio = rocblas_ms / persistent_ms`，> 1 是我们更快。只看自己加 `DPA4_ROCBLAS=0`；
rocBLAS 包装编不过时用 `DPA4_ROCBLAS_CPP_VERBOSE=1` 看原因（它不会让测试变红）。

## 6. Profile（可选）

`docs/dmk-profiling.md`。关键提醒不变：持久化 GEMM **只有一次 kernel launch**，rocBLAS 是一串。

## 7. 一键 1→5

```bash
set -e
cd ~/private_data/lipengfei/bysj && source /opt/dtk/env.sh
export PYTORCH_ROCM_ARCH=gfx928          # 或 DPA4_GEMM_SCALAR=1
MAX_JOBS=8 python3 setup.py build_ext --inplace
python3 tests/run_python_tests.py
PYTHONPATH=python python3 -m unittest tests.csrc.accuracy.test_persistent_gemm_smoke -v
PYTHONPATH=python python3 -m unittest tests.csrc.accuracy.test_persistent_gemm -v
DPA4_RUN_PERSISTENT_GEMM_LATENCY=1 PYTHONPATH=python \
  python3 -m unittest tests.csrc.latency.test_persistent_gemm_latency -v
```

## 判读：性能数字说明什么

| 观测 | 含义 | 下一步 |
| --- | --- | --- |
| 标量分支只有 1–3 TFLOP/s | 就是标量上限（`fma_probe` 18.1 是理论峰值） | 要么换卡，要么接受 |
| DUMMA 分支低于 rocBLAS 的 0.3× | 单波 A 复制 4 份、无 L2 分块、无宏 tile | 宏 tile + 共享 A 页 |
| 大形状反而更差 | 访存放大（A 被重读 `N/16` 次） | 改 tile 顺序 / 加 L2 分块 |
| 小形状（512³）特别差、大形状追近 | launch 与页开销占比高 | 一条指令覆盖更多 tile |
