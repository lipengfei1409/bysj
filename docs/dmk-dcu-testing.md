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

**默认目标架构是 `gfx928`**（部署卡），不是"编译机是什么就编什么"：

```python
DEFAULT_ARCH = "gfx928"
ARCH = os.environ.get("PYTORCH_ROCM_ARCH") or DEFAULT_ARCH
```

于是三种情形：

| 情形 | 行为 |
| --- | --- |
| 编译机就是 gfx928 | 直接用，DUMMA 路径可用 |
| 编译机是别的卡（如 gfx906） | 照 `gfx928` 编，并打印 `note: building for gfx928; this device is gfx906 ... will not run here`；**产物只能在 928 上跑** |
| 想在本机跑 | `export PYTORCH_ROCM_ARCH=gfx906`，但 gfx906 没有矩阵单元 → 必须再加 `DPA4_GEMM_SCALAR=1` |

`gfx906` 上还有一个重要事实：**rocBLAS 的 SGEMM 在 gfx906 上同样是 VALU/FFMA 实现**（这张卡没有
MFMA/DUMMA）。所以那边的对比是"同样用标量单元，谁的调度更好"，不是"标量 vs 矩阵单元"——差距会落在
2–4 倍的量级，而不是 10 倍以上。

本机到底是哪张卡，以 `rocminfo` 为准：

```bash
rocminfo | grep -m1 amdgcn-amd-amdhsa         # → amdgcn-amd-amdhsa--gfx928:...
```

`torch.cuda.get_device_properties(0).gcnArchName` **不可靠**：在 gfx928 的机器上曾报成 `gfx906`，
把 DUMMA 路径误排除了。`setup.py` 现在两者都问，冲突时以 `rocminfo` 为准并打印一句说明。

**为什么只编一个架构**：`PYTORCH_ROCM_ARCH` 没设时，torch 会按套件默认编**全部**已知架构（早先那次
就是 gfx906+gfx926+gfx928 三个一起编），每个架构都要完整跑一遍 device 编译；而且只要其中有一个不受支持
的架构，整个构建就失败。`setup.py` 因此在未显式设置时取**本卡架构**这一个。想编多个（要在多张不同卡上
跑）就显式给：

```bash
export PYTORCH_ROCM_ARCH="gfx906;gfx928"     # 逗号或空格分隔也认
```

显式值永远优先于自动探测；但开 DUMMA 时列表里**每个**架构都必须是 gfx926/928/936/938 之一。

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

**换架构、改头文件、或"报的错和上次一模一样"时，先清干净**：

```bash
python3 tools/clean_build.py --dry-run   # 先看会删什么
python3 tools/clean_build.py             # build/ + hipify 产物 + 就地扩展
MAX_JOBS=8 python3 setup.py build_ext --inplace
```

> 清理是**独立脚本**而不是 `setup.py clean` 的子类：在 `setup.py` 里继承 distutils 的
> `clean` 会在 import 阶段把 setuptools 自带的 distutils 拉进来，某台机器上直接让 `setup()`
> 挂在 `MinimalDistribution has no attribute set_defaults`。构建工具进不去，比多一条命令更糟。

它会删三类残余，各有各的坑：

| 残余 | 为什么必须删 |
| --- | --- |
| `build/`（ninja 对象、临时目录） | 换架构后旧对象不会自动失效 |
| `csrc/**/*_hip.cpp` / `*_hip.h` | **torch 的 hipify 把产物写在源码目录里**，陈旧的副本会被当成真实头文件编译 |
| `python/kernels/_C*.so` | `--inplace` 会覆盖它，但 `clean` 原本不管 |

编译器的完整输出写到 **`build/build.log`**，终端只打印错误行和日志路径：

```
build failed: 3 error line(s)
  /…/csrc/ops/gemm.h:190:21: error: implicit instantiation of undefined template …
  /…/csrc/dmk/state.h:51:31: error: call to function 'make_op' that is neither …

full output: /…/build/build.log
```

成功的构建只打印一行 `build ok -- compiler output: …/build/build.log`。
一次构建会有 800+ 条警告（几乎全来自 torch / pybind 头），所以把错误挑出来是省时的关键——尤其
每轮修复都要上卡的时候。

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
| 16×64 | 两条指令各覆盖两个 tile：两个 consumer wave + 两个 C 页 + storer 遍历 |
| 16×48 | 最后一个 run 只有三个 tile：**wave 3 必须完全不动**（越界裁剪） |
| 64×256 | 四行 tile，每行一个满 run |
| 128³ | 深 K：每波跨 8 个 k-tile 累加 |

失败时的分流：

| 失败在 | 说明 |
| --- | --- |
| `16³` | `ld_matrix` / `store_matrix` 的参数（`ld`、`mem_row_major`）或累加器交替 |
| `16³` 过、`16×32` 错 | 指令覆盖两个 tile 的 `n0` 计算（主机 `GemmEmitter.emit`） |
| 小形状过、`16×64` 错 | 多波并行：`pid(2 + wave)` 的 C 页分配或 `tile_column` |
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
