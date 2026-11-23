# 把 fused SO(2) 卷积移植到 HIP（`csrc/so2_conv_hip/`）

目标：让 DeepMD 自己的融合算子 `deepmd::dpa4_so2_conv` 在 DCU 上可用。
**不修改 DeepMD 源码**——翻译后的算子注册进**同一个 `deepmd` 算子命名空间**，
于是上游的 Python 侧（`op_available()`、`_setup_context`、`_backward`、fake impl、
`_profitable`）原样生效。

- 上游实现：`deepmd-kit/source/op/pt/dpa4/so2_conv{,_kernel}.cuh` + 24 个 `(CF,L)` 特化
- 本目录：机械翻译 + 4 处人工修补，实例化全部 12 个 `(CF, L)` 对 × 两个方向（24 个 TU）
- 为什么值得做：算术投影 **55 ms** vs 稠密前向 **163.8 ms**（见 [`execution-model.md`](execution-model.md) §5）

## 1. 目录

| 文件 | 内容 |
| --- | --- |
| `tools/hipify_so2_conv.py` | **生成器**：copy → `torch.utils.hipify.hipify` → 4 处人工 patch → lint。改代码改这里 |
| `hip_compat.cuh` | 唯一手写的 shim：32-lane 单元、divergent barrier |
| `so2_conv.hip` | 主机侧：校验、CSR、op schema、`dpa4_fp32_ridge`（重写为 HIP 版） |
| `so2_conv_hip.cuh` | 布局代数 + warp 归约 |
| `so2_conv_kernel_hip.cuh` | 前向/反向 kernel 本体（1112 行） |
| `so2_conv_launch_hip.h` | 启动策略与 `(L,CF)` 分派 |
| `so2_conv_instantiate_hip.cuh` | 模板实例化收尾 |
| `so2_conv_{fwd,bwd}_c{32,64}_l{1..6}.hip` | 24 个特化（各 10 行） |
| `build.py` | 自己驱动 `hipcc` 编译，再 `torch.ops.load_library` 注册（见 §4.1） |

`hipify` 的产物是自动生成的，**不要手改**；改 `tools/hipify_so2_conv.py` 后重跑：

```bash
python3 csrc/so2_conv_hip/tools/hipify_so2_conv.py   # 任何有 torch 的机器
```

## 2. 翻译映射

| CUDA | HIP | 说明 |
| --- | --- | --- |
| `cuda_runtime.h` / `ATen/cuda/*` | `hip/hip_runtime.h` / `ATen/hip/*` | hipify 自动 |
| `cudaStream_t`、`cudaXxx`、`cudaDevAttr*` | `hipStream_t`、`hipXxx`、`hipDeviceAttribute*` | hipify 自动 |
| `k<<<g,b,s,st>>>(args)` | `hipLaunchKernelGGL((k), g, b, s, st, args)` | hipify 自动；它会把 `dim3` 套两层，已归一化 |
| `__shfl_xor_sync(mask, v, d)` | `__shfl_xor(v, d, kWarp)` | **人工**，`kWarp = 32`，见 §3.1 |
| `__syncwarp()` | `dpa4_wave_barrier()` → `__builtin_amdgcn_wave_barrier()` | **人工**，见 §3.2 |
| `__pipeline_memcpy_async` / `_commit` / `_wait_prior` | `__builtin_memcpy` / 注释 / 注释 | **人工**，见 §3.3 |
| `dpa4_fp32_ridge()`（读 CUDA compute capability、128 lane） | `hipGetDeviceProperties` + `prop.warpSize` | **人工**，见 §3.4 |

## 3. 四个必须人工处理的点

### 3.1 32-lane 单元是算法的一部分，不是习惯

上游 kernel 的每一列输出都映射成 `j * 32 + lane`，`kWarp = 32` 切分通道维
（`CFB = CF / 32`）。ROCm 的 wavefront 是 64 lane，所以**保留 32-lane 单元、一个 wave 里跑两个**
——`__shfl_xor(v, d, kWarp)` 的 `width = 32` 让交换只在每个 32-lane 段内进行，语义与 CUDA 一致。
代价是波前占用率减半。**把宽度改成 64 是改算法，不是移植**，不在 v1 范围。

### 3.2 `__syncwarp()` 在发散代码里

四处 `__syncwarp()` 都在 `if (active) { … }` 内部（发布 warp 私有 stage、读回广播值）。
换成 `__syncthreads()` 会死锁；AMD 的 wave barrier 是保守超集（顺带同步同 wave 的另一半，
无害且更严格）。

### 3.3 异步 pipeline 变成同步

`__pipeline_memcpy_async` 每次拷 4 个 float（16 B），双缓冲让权重面板的搬运与计算重叠。
v1 直接换成 `__builtin_memcpy`，两个 wait 变成注释——因为前后各有一个 `__syncthreads()`
已经为共享缓冲排序，正确性不受影响，只丢了重叠。代码里留了 `v1: synchronous stage`
标记，v2 换 `hip::memcpy_async` 时按它改。

### 3.5 ATen 的 masquerading 头必须换成 c10（编译才会过）

hipify 会把 `ATen/cuda/CUDAContext.h` 变成 `ATen/hip/HIPContext.h`，而这个头会拉进
`ATen/hip/Exceptions.h`：

```
/usr/local/.../torch/include/ATen/hip/Exceptions.h:10:10: fatal error:
      'cusolver_common.h' file not found
```

`cusolver_common.h` 是 CUDA 的，ROCm/DTK 里没有。当前流在 c10 层有对应物、也不拉求解器头：
`c10/hip/HIPStream.h` 的 `c10::hip::getCurrentHIPStream()`。

**设备守卫干脆不用 torch 的**：它的头位置在每个 torch 构建里都不一样——这台机器上没有
`c10/hip/impl/HIPGuardImplMasqueradingAsCUDA.h`（第一次改就撞上），另一台可能是
`ATen/hip/impl/...`，第三台可能两处都没有。`hipGetDevice`/`hipSetDevice` 永远在，
所以生成器注入一个 20 行的 `dpa4::DeviceGuard` RAII 包装，行为等价（torch dispatch 通常
已经选好设备，这层只是保险）。

### 3.6 分派必须只引用"已编译的 shard"

第一次链接成功后 `load_library` 报：

```
undefined symbol: dpa4::conv_forward_launch<1, 32>(...)
```

原因是上游的 `DPA4_CONV_FOR_EACH_SHAPE` 展开全部 **12 个 `(CF, L)`** 组合，
而 v1 只编译 `(3, 64)`，于是分派函数引用了 11 个不存在的符号——链接不报错（它们是
`extern template` 声明），运行时 dlopen 才炸。生成器因此注入
`DPA4_CONV_SHIPPED_SHAPES(macro)`（由 `SHARDS` 列表生成），分派与
`conv_shape_instantiated()` 都改用它。主机测试会核对：分派里出现的每个 `(L, CF)`
都必须在 `build.SOURCES` 里同时有 fwd 与 bwd 的 shard。

### 3.4 `dpa4_fp32_ridge()` 必须重算

它决定 `_profitable` 的设备缩放，原实现读 CUDA 的 compute capability 和“128 FMA lane”：
在 AMD 上是错的。HIP 版用 `hipDeviceProp_t`：`2 × warpSize × CU × 时钟`（CDNA 的 fp32 FMA
宽度就是波前宽度）。

**实测：`dpa4_fp32_ridge() = 24.0`**（标定件 73.2），与 FFMA 探针折算的 28.7 同量级——
两者口径不同（这里用理论峰值和理论带宽，探针用实测持续吞吐），差 20% 属正常。
意义：`_profitable` 的预算 = 65536 × 24.0/73.2 = **21 490 FMA**，而本 checkpoint 需要
**225 280**，差 10.5 倍 → 上游会拒绝，接线时必须显式旁路。

## 4.1 为什么不用 `torch.utils.cpp_extension.load`

第一次在 DCU 上构建就踩到了：

```
File ".../cpp_extension.py", line 2799, in object_file_path
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

ROCm 上 `load()` 会对每个源文件跑一遍 hipify，然后用
`hipify_result[s].hipified_path` 组对象列表；**对已经是 `.hip` 的源文件这个值是 `None`**，
于是 `sources` 里混进 `None`，ninja 初始化时 `basename(None)` 炸掉。

而且这个 hipify 本来就是多余的：这里的源文件已经是我们人工 patch 过的 HIP 代码，
再跑一遍只会自找麻烦。所以 `build.py` 改为**直接调 hipcc**：

```python
hipcc -c src.hip -o obj.o -O3 -std=c++17 -fPIC -fno-gpu-rdc --offload-arch=gfx928 \
      -D_GLIBCXX_USE_CXX11_ABI=<与 torch 一致> -D__HIP_PLATFORM_AMD__ -I<torch include> -I<port>
hipcc -shared -o dpa4_so2_conv_hip.so *.o -L<torch lib> -ltorch -ltorch_hip -lc10
torch.ops.load_library(so)
```

- `--offload-arch` 从 `torch.cuda.get_device_properties(0).gcnArchName` 取，去掉 `:sramecc+` 之类后缀；
- ABI 标志从 `torch._C._GLIBCXX_USE_CXX11_ABI` 取，不一致会加载失败；
- 产物落在 `~/.cache/dpa4_so2_conv_hip/`（可用 `DPA4_SO2_CONV_HIP_BUILD` 覆盖），
  源码或头文件更新后自动重编，`--clean` 强制全编；
- `--dry-run` 只打印编译命令，**在没卡的机器上也能看**（先过 HIP 检查会提前退出）。

## 4. 构建与验证（DCU）

```bash
cd ~/private_data/lipengfei/bysj
source /opt/dtk/env.sh
git pull

python3 csrc/so2_conv_hip/build.py --check
```

期望输出：

```
registered: True
  deepmd::dpa4_so2_conv: ok
  deepmd::dpa4_so2_conv_backward: ok
  deepmd::dpa4_fp32_ridge: ok
default schema:
  dpa4_so2_conv(Tensor x, Tensor src, ...) -> (Tensor out, Tensor alpha, Tensor pre_gate, Tensor z_all)
dpa4_fp32_ridge(): 28.7
```

`registered: True` 之后，DeepMD 自己的 `op_available()`（`pt_expt/kernels/cuda/dpa4/so2_conv.py`）
会返回 True，`make_cuda_so2_conv` 就会去构造入口——但**还需要过 `_is_supported` 和 `_profitable`**。

## 4.5 单算子测试：直接跑上游自己的测试

上游为这个算子带了一整套参考测试（`source/tests/pt/model/test_descriptor_sezm_cuda.py`
的 `TestSeZMConvCuda` 与 `TestSeZMConvCudaGate`）：**每个 zoo 形状的前向 vs 稠密转录**、
**反向 vs autograd**、归约可复现、`make_fx` 追踪、零边/零 envelope、多头拒绝。
它们以 `op_available()` 为门，所以在收集测试之前把本移植注册进去即可全部生效——
这正是 `tests/dcu/so2_conv_hip_plugin.py` 做的事（pytest 插件，收集前 `build()`）。

```bash
DEEPMD_SRC=~/private_data/lipengfei/deepmd-kit \
  python3 tests/dcu/run_upstream_conv_test.py
```

它先 `build.py`（预热扩展缓存、顺带打印 schema），再用 `-k ConvCuda` 只跑卷积那几组。
第一次在 DCU 上跑，这一条命令同时回答"编得过吗"和"数值对吗"。

## 5. 已知取舍（v1）

| 取舍 | 影响 | 何时修 |
| --- | --- | --- |
| 只实例化 `(lmax=3, CF=64)` | 只服务本 checkpoint | 需要时在生成器的 `FILES` 里加 shard |
| 32-lane 单元 → 半波前占用 | 有效吞吐上限约 50% | 需要真正提速时改成 64-lane 算法 |
| 同步 staging | 权重面板搬运不与计算重叠 | 实测确认是瓶颈后 |
| 未接 `_profitable` 旁路 | 上游仍会拒绝本形状（225 280 vs 25 718） | 下一步：`deepmd_patch` 里加显式开关 |

## 5.5 单算子测试结果（2026-11-23）

```
$ python3 tests/dcu/run_upstream_conv_test.py
library: /root/.cache/dpa4_so2_conv_hip/dpa4_so2_conv_hip.so
registered: True
10 passed, 16 deselected, 55 subtests passed in 10.00s
```

这一结果同时验证了三处保守处理都是对的：

1. **32-lane 单元保留**：`__shfl_xor(v, d, kWarp)` 在 64-lane wave 上按段交换，前向与稠密转录逐形状一致；
2. **发散区的 wave barrier**：`if (active)` 内的四处 `__syncwarp()` 换成 `__builtin_amdgcn_wave_barrier()` 后，
   归约可复现测试（`test_reduction_is_reproducible`）通过；
3. **同步化的权重 staging**：双缓冲被去掉后反向仍与 autograd 一致，说明前后两个 `__syncthreads()`
   确实排好了序。

**未验证**：性能。算子算得对不等于算得快，下一步是端到端 A/B。

## 6. 接线（已实现）

`python/model/so2_conv_hip.py` 把两道门打开，`DPA4Model(..., so2_conv="hip")` 与
profile CLI 的 `--so2-conv hip` 都走它：

1. 加载扩展（`csrc/so2_conv_hip/build.py` 的 `build()`）；
2. 在**构造 `DeepPot` 之前**把 `DP_CUDA_INFER` 抬到 2（`cuda_infer_level()` 在模型构造时读取）；
3. 把 `deepmd.pt_expt.kernels.cuda.dpa4.so2_conv._profitable` 替换为恒真，并在
   `status()` / summary 的 `so2_conv` 字段里**显式记录**这个覆盖——它是工程判断
   （基准是稠密路径而非 Triton），不是上游的推荐；
4. `disable()` 还原判据与环境变量（CLI 进程内保持开启，因为 force 路径每次调用都会重读该门）。

前置条件都满足后才允许：`registered` 为假时 `enable()` 返回原因、不碰判据，
`DPA4Model` 直接抛 `RuntimeError` 而不是静默退回稠密。

## 6.4 结论：数值正确，性能不合适 → 封存（2026-11-24）

四组可编译的 tile 全部扫完（`BE = WARPS·TM` 是权重复用率，`smem` 决定驻留数）：

| tile `(TM, WARPS, PK, OCC)` | smem | 驻留 | 前向 | 反向 | 合计 | 相对最好 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4, 4, 8, 2`（默认） | 45.0 KB | 1 | 474.0 | 878.8 | 1352.8 | 1.57× 慢 |
| **`2, 4, 4, 2`** | **24.2 KB** | **2** | **381.8** | **481.7** | **863.5** | — |
| `4, 2, 4, 2` | 17.4 KB | 3 | 594.4 | 787.0 | 1381.4 | 1.60× 慢 |
| `1, 2, 4, 2` | ~19 KB | 3 | 1278.4 | 1444.8 | 2723.2 | 3.15× 慢 |

最好的一组 `863.5 ms`，而它替换掉的稠密工作约 **690 ms**（由 957.0 → 1621.6 减去 conv 自身反推），
即 **0.80×**。两个方向都已顶到约束：

- 再压 smem 只能减 `WARPS`，但每 block 的并行度随之下降（`4,2,4,2` 反而慢 60%）；
- 提复用要增 `TM`，但寄存器随 `TM·RB` 上升，`TM=4` 已是 `RB=20` 的上限。

**这与上游自己的判据一致**：`_profitable` 用 ridge 归一化，标定件 73.2、本卡实测 24.0，
而 H20（ridge 更低）上对 `air` 形状的实测就是 0.79×。我们得到 0.80×，两条曲线对上了——
说明**结论不是"移植做错了"，而是"这块卡的 ridge 落在不划算的一侧"**。

因此：代码保留（数值正确、可作为 v2 的基线），默认 `--so2-conv off`，**不再在 tile 上投入**。
若将来换到 ridge 更高的卡（fp32 更强或带宽更低），这条路可以直接复测。

## 6.4 收益怎么读

只看一个数：**`device ms`（`dpa4/inference.single_energy_force` 注解的设备时间）**。
三组 A/B 用一条命令读：

```bash
python3 tests/deepmd/profile/ab_report.py "$OUT" base kc3 kc8
```

```
tag       device ms    delta   conv ms  launches  routed
base          957.0     +0.0     863.5        217  on
kc3           880.0    -77.0     620.0        217  on

-- verdict --
  kc3: -77.0 ms (1.087x) -- net win
```

判据（按信任度排序）：

1. **`delta < -20 ms` 才算数**：这套 harness 的 run-to-run 抖动约 20 ms（957.0 vs 964.4 是同一配置的两次）。
2. **`conv ms` 必须低于 ~690 ms**：那是它替换掉的稠密工作量；conv 自己更快但总量不动，说明省下的时间被别处吃掉了。
3. **`routed` 必须是 `on`**：为 `off` 时脚本会写明原因（扩展没编上 / 没有可见 DCU），
   不要把"没走这条路"读成"没有收益"。

`ab_report.py` 把这些判断写成了代码，`--json` 会落盘，方便贴进报告。

## 6.45 继续压：把 `kc` 移出共享内存（`DPA4_KC_GLOBAL`）

smem 账里最大的一块是可移走的：`kc = BE·kc_len = 8·256 fl = 8 KB`。它在 global 里按边连续
（每边 1 KB）、L2 常驻，只是被 `DegreeMixer::kernel(slot)` 反复读，所以**可以不入 smem**：

```cpp
#if DPA4_KC_GLOBAL
  kc = nullptr;                        // bind()
  alpha = wig + T::BE * S::NW;         // alpha 前移
#else
  kc = wig + T::BE * S::NW;
  alpha = kc + T::BE * kc_len;
#endif
```

配套三处：`bytes()` 不再计 `kc_len`、`load_chunk` 跳过拷贝、`mixer_of` 多接一个真实边号
（`mix.kc = a.kc + edge * a.kc_len`）。默认 `0`，已验证的二进制不变。

**实测（2026-11-24，`ab_report.py`）**：

| 变体 | device ms | delta | conv ms | 结论 |
| --- | ---: | ---: | ---: | --- |
| `TM=2 PK=4`（基准） | 1126.2 | — | 864.2 | 0.80× 稠密 |
| `TM=2 PK=4 KC_GLOBAL` | 1101.6 | **−24.6** | 834.4 | 只有 3.4%，**驻留不是杠杆** |
| `TM=2 PK=8 KC_GLOBAL` | 1382.9 | **+256.7** | 1122.1 | 深面板反而慢 30% |

读数：

- 若驻留真从 2 块升到 3 块，按 1→2 块时那 1.57× 的斜率，这里应看到 1.3× 以上；
  实测 1.03×，说明**限制不在 smem**——要么寄存器把第三块卡住了（`OCC=2` 时
  `__launch_bounds__` 允许每线程 256 个 VGPR，编译器很可能就用满了），要么多出来的
  global 读正好抵掉。
- `PK=8` 少了**一半**屏障却慢 30%，说明真正的成本是**每次面板拷贝的延迟被完全暴露**
  （同步 staging），面板越大暴露越多。

所以下一个、也是**最后一个**有机制依据的杠杆是 ② 寄存器中转的双缓冲 staging；
它的上限是"打平"（conv 需要从 834 降到 ≤690，即再快 17%），不是"大胜"。

## 6.46 共享内存 + 寄存器双缓冲（`DPA4_STAGE_OVERLAP`）

移植时把上游的 `__pipeline_memcpy_async` 换成了同步 `__builtin_memcpy`，于是每次面板拷贝的
全局 load 必须在下一次屏障前**完成**，延迟完全暴露。可移植的恢复办法是寄存器中转：

```cpp
float4 prefetch[VPT];                    // 每线程一个面板的切片
load_panel(panel + 2);                   // 在 consume 之前发起下下块
__syncthreads();
store_panel(panel + 1);                  // 把上一轮预取写进另一块 smem
__syncthreads();                         // 发布，然后 consume 当前块
```

**屏障数不变（仍是 2 个）**，只是全局 load 与 `consume` 重叠。`VEC == 1`（窄 focus width）
自动退回同步路径，所以 12 个 `c32` 特化照旧编译。

默认 `0`，可与 `KC_GLOBAL`、`PK` 自由组合：

```bash
# 只加重叠（保持 2 块驻留）
DPA4_SO2_CONV_HIP_DEFINES="DPA4_TILE_TM=2 DPA4_TILE_PK=4 DPA4_STAGE_OVERLAP=1" \
  python3 csrc/so2_conv_hip/build.py --check --full
# 重叠 + 深面板（PK=8 此前因暴露延迟而慢 30%，重叠正是它缺的东西）
DPA4_SO2_CONV_HIP_DEFINES="DPA4_TILE_TM=2 DPA4_TILE_PK=8 DPA4_STAGE_OVERLAP=1" \
  python3 csrc/so2_conv_hip/build.py --check --full
```

## 6.5 第一次端到端的结果：慢 2.6 倍（2026-11-24）

```
device total : 957.0 ms -> 1621.6 ms (0.590x)
  dpa4::so2_conv_fwd_kernel<3,64,ConvTile<4,4,8,2>>   474.0 ms
  dpa4::so2_conv_bwd_kernel<3,64,ConvTile<4,4,8,2>>   878.8 ms
```

数值对、但比稠密慢。**主因是 occupancy，不是 lane 空转**（这一点我先前的说法是错的，已更正）：

按 `ConvSmem` 的布局算 (L=3, CF=64)：

| 项 | 浮点 | KB |
| --- | ---: | ---: |
| `SCRATCH = max(PANEL, OUTER·WARPS)`，PANEL = 2·PK·NMAX = 6144 | 6144 | 24.0 |
| `kc = BE·kc_len` = 16·256 | 4096 | 16.0 |
| `astage + wig + alpha + rescale` ≈ | 1264 | 4.9 |
| **合计** | | **45.0** |

CDNA 的 LDS 是 **64 KB/CU**，45 KB 只装得下 **1 个 block**（策略却要 `OCC = 2`）。
一个 block 是 128 线程 = 2 个波前，而 `fma_probe` 的 18.1 TFLOP/s 是在 960×256 的高占用下测的——
**占用率只有百分之几时，SIMT 吞吐会掉到峰值的 1/4–1/8**，这正是 474 ms 对 55 ms 理想值的差距。

32-lane 单元本身**不会让 lane 空转**（NT=128 是 64 的倍数，波前被填满）；它的代价只是
barrier 粒度与寄存器归属，不是吞吐。同步化的 staging 是次要项（§3.3）。
4. `TM=4 / WARPS=4 / PK=8 / OCC=2` 是在 RTX PRO 6000 与 H20 上测出来的。上游自己写明：
   *"What must be retuned per part … TM thresholds: the 64 K-register file; PK: shared memory per
   block against barrier count; OCC: the register file again"*，并且留了 `-DDPA4_TILE_<name>=<value>`
   的 sweep 入口。

**只有 `WARPS` 能真正把 smem 压下来**（它同时缩小 `kc`、`astage`、`wig`、`OUTER`）；
`PK` 单独降到 2 只剩 34.4 KB，仍然只装 1 个 block —— 这就是 `PK=2` 那次 A/B 没动的原因。
按 `ConvSmem` 算出的驻留数：

| 变体（PK 必须是 4 的倍数） | smem | 每 CU 可驻留 |
| --- | ---: | ---: |
| 默认 `WARPS=4 TM=4 PK=8` | 45.0 KB | **1** |
| `TM=2 PK=4` | 24.2 KB | 2 |
| `WARPS=2 PK=4` | 22.5 KB | 2 |
| `TM=2 WARPS=2 PK=4` | 22.5 KB | 2 |
| `TM=1 WARPS=2 PK=4`（未验证 TM=1 是否编译） | ~19 KB | 3 |

`PK` 只能是 **4 的倍数**：`so2_conv_kernel_hip.cuh` 里
`static_assert(T::PK % 4 == 0, "panel depth must cover the packed step group")`，
而 staging 还要 `kWarp % T::PK == 0`。我第一次建议 `PK=2` 就撞在这条上——
24 个 TU 全挂、报错还都是同一行。现在 `build.py` 会在编译前拒掉非法组合并给出原因，
编译失败时也会带上第一条 `error:` 而不是只给一个 exit code。

**第一次有效扫描（2026-11-24）**：`DPA4_TILE_TM=2 DPA4_TILE_PK=4` → 24.2 KB / 2 块驻留，
kernel 名变成 `ConvTile<2, 4, 4, 2>`：

| kernel | 默认 tile | TM=2 PK=4 | 变化 |
| --- | ---: | ---: | ---: |
| `so2_conv_fwd_kernel` | 474.0 ms | **381.8 ms** | −19% |
| `so2_conv_bwd_kernel` | 878.8 ms | **481.7 ms** | −45% |
| 合计 | 1352.8 ms | **863.5 ms** | −36% |

**occupancy 假设成立**：驻留从 1 块升到 2 块，整段快了 1.57×。但 863.5 ms 仍高于它替换掉的
稠密工作（约 690 ms，由第一次 A/B 的 957 → 1621.6 减去 conv 自身反推），即 **0.80×，还没打平**。
离打平还差约 1.2×，理论上需要 3 块驻留。

**第二次扫描失败（2026-11-24）**：`TM=1 WARPS=2 PK=4` → `ConvTile<1, 2, 4, 2>`，
前向 **1278 ms**、反向 **1445 ms**（合计 2723 ms，比默认 tile 还差一倍）。

原因不是 smem，而是 **`BE = WARPS·TM` 决定了权重面板的复用率**：每块权重装载要喂 `TM` 个乘积，
`TM=1` 把复用砍到 1/4，权重流量按 FLOP 翻了四倍。上游注释说的就是这件事——
*"the activation tile costs TM·RB registers per thread and every weight load feeds TM products,
so TM trades occupancy against weight traffic"*。

于是约束变成两条，互相耦合：

- **`BE = WARPS·TM ≥ 8`**（权重复用不能塌，`TM=1` / `WARPS=1` 直接出局）；
- **smem < 21.3 KB**（要 3 块驻留）。

在 BE=8 的前提下压 smem，只能减 `WARPS`（`OUTER ∝ WARPS`），所以唯一还没试的候选是
**`TM=4 WARPS=2 PK=4`**：BE 仍为 8，`OUTER` 从 13.7 KB 降到 6.9 KB，合计约 **17.4 KB → 3 块**。

```bash
export DPA4_SO2_CONV_HIP_DEFINES="DPA4_TILE_TM=4 DPA4_TILE_WARPS=2 DPA4_TILE_PK=4"
```

这是最后一组：若仍不及 `TM=2 WARPS=4 PK=4` 的 863.5 ms，就把 conv 封存在"数值正确、性能不合适"，
默认 `--so2-conv off`。

sweep 的做法（define 必须在**运行**时也生效，所以用环境变量而不是只传给 `build.py`）：

```bash
export DPA4_SO2_CONV_HIP_DEFINES="DPA4_TILE_WARPS=2 DPA4_TILE_PK=2"
python3 csrc/so2_conv_hip/build.py --check                 # 建到 dpa4_so2_conv_hip_DPA4_TILE_WARPS2-DPA4_TILE_PK2/
python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 ... \
  --fused on --so2-conv hip --output-dir "$OUT/e2e_w2pk2"
```

每一组 define 落在独立缓存目录，可与默认构建并存。**验证是否真的换了 tile**：trace 里
kernel 名会从 `ConvTile<4, 4, 8, 2>` 变成 `ConvTile<4, 2, 2, 2>`（TM, WARPS, PK, OCC）。
那次 `PK=2` 的 A/B 名字没变，说明二进制没换——这是当时没看出区别的第一个原因。

## 7. 验证清单

| 检查 | 在哪 | 状态 |
| --- | --- | --- |
| 翻译无 CUDA-only token | `tests/python/test_so2_conv_hip_port.py`（主机） | ✅ |
| `kWarp` 仍是 32、shuffle 宽度 32 | 同上 | ✅ |
| 发散 barrier 用 wave 内建 | 同上 | ✅ |
| schema 与上游逐字一致 | 同上（C 字面量拼接后比对） | ✅ |
| ridge 用 HIP 属性重算 | 同上 | ✅ |
| **能编译、能注册** | DCU：`python3 csrc/so2_conv_hip/build.py --check` | ✅ 2026-11-23，5 个算子全部 ok，ridge=24.0 |
| **数值与稠密一致** | DCU：`tests/dcu/run_upstream_conv_test.py` | ✅ 2026-11-23：**10 passed / 55 subtests**（前向 vs 稠密逐 zoo、反向 vs autograd、归约可复现、make_fx、零边、零 envelope、多头拒绝） |
| **接线** | `--so2-conv hip` / `DPA4Model(so2_conv="hip")` | ✅ 主机 6 项测试 |
| **端到端收益** | DCU：`--so2-conv hip` 的 A/B | ⬜ 待跑 |
