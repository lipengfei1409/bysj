# DCU DPA4

面向 ROCm/DTK 的 DPA4 算子与独立建模工程。项目包含 DeePMD eager adapters 和 `myproject` 自有数值路径；后者正在按 DPA4-Neo OMol25 checkpoint 补齐。**当前仍不支持完整独立 E/F/V，也没有经 DCU 验证的 MegaKernel 加速结果。**

## 目录

```text
myproject/
├── include/              header-only DCU 指令/类型层（不依赖 Torch）
│   ├── blas/             host rocBLAS GEMM/batched封装与row-major布局转换
│   ├── common/           config、matrix view、thread/shared tile
│   └── ops/              thread FMA/reference MMA；group DUMMA/memory/wave/sync
├── csrc/                 HIP 算子、C++ 检查与 pybind 接口
│   ├── parallel/         geometry / radial / norm / scatter / force
│   ├── gemm/             linear 正确性基线
│   └── include/
├── python/
│   ├── dcu_dpa4/         算子合同、Torch reference、HIP dispatch/autograd
│   │   └── deepmd/       DeePMD 实例级 adapters、eager 配置与安装/恢复
│   └── model/            checkpoint、数据、reference runner、modeling
├── tests/
│   ├── csrc/             native 精度、一阶梯度、stream 测试和 benchmark
│   └── python/           Python 接口、shape、checkpoint 和功能测试
└── doc/                  数据流、shape、DTK 实现约定与后续计划
```

文档：[MegaKernel 架构](doc/dpa4-megakernel-architecture.md)、[算子合同](doc/dpa4-operators.md)、[DTK 开发约定](doc/dtk-native.md)、[数值模型](doc/numerical-modeling.md)、[DeePMD 接入](doc/deepmd-integration.md)、[rocBLAS](doc/blas.md)、[DUMMA 指令层](doc/instructions.md)、[验证记录](doc/verification.md)。

独立模型的 checkpoint/API 和当前实现边界见 [数值模型说明](doc/numerical-modeling.md) 与 [MegaKernel 架构说明](doc/dpa4-megakernel-architecture.md)。当前前端与部分 SO(3)/SO(2)/attention 算子已有代码；OMol25 的 message-node/FFN grid、readout/fitting 等活跃路径仍缺失，`complete_model` 保持 false。

## 指令层

BLAS库封装另见 [rocBLAS说明](doc/blas.md)：`include/blas/rocblas.hpp` 为host库调用，Python入口为 `dcu_dpa4.blas.gemm/linear`；可选构建 `DPA4_BUILD_HIP=1 DPA4_BUILD_ROCBLAS=1`。它是库级基线，不是device MMA指令，当前不自动替换原linear。

手册 API 的独立封装在 `include/ops/group/dumma.cuh`：`dcu::dumma::fill_fragment/load_matrix_sync/mma_sync/store_matrix_sync` 一一对应 DTK DUMMA 第 2 节，直接复用 `du_mma.h` 的原生 fragment 和函数；固定 shape 便捷类也通过此层调用。

统一包含 `#include "ops/ops.cuh"`，命名空间为 `dcu`；参考 ThunderKittens 按类型和作用域分层，但使用 HIP/DTK 后端。接口、MMA shape 和实机测试命令见 [指令层说明](doc/instructions.md)。`thread::mma_reference` 是标量 reference；`group::mma_f32_16x16x4/8`、`group::mma_f16_16x16x16` 才是 DUMMA 硬件封装，需要显式 `-DDPA4_ENABLE_DUMMA=1` 与兼容的 DTK/gfx 架构。当前不改变 linear 默认执行路径。

## 接入 DeePMD-kit（推荐数值路径）

无需改 DeePMD 源码；详见 [Python 接入说明](doc/deepmd-integration.md)。在已有 DeePMD 的 Python ≥3.10 环境中：

```python
from dcu_dpa4.deepmd import configure_eager, install_operators
configure_eager("cpu")  # DCU 环境为 "dcu"
from deepmd.infer import DeepPot
dp = DeepPot("pretrained.pt", no_jit=True)
with install_operators(dp.get_model(), backend="reference") as handle:
    e, f, v = dp.eval(coord, cell, atype)
    print(handle.report())
dp.close()
```

首批可替换 radial、普通 RMSNorm、MLPLayer/ChannelLinear/nn.Linear。其他模块执行 DeePMD 原实现；参数对象与 state_dict 不变，卸载后恢复原 forward。HIP backend 无隐式 fallback，仅支持 FP32 eval 推理，不支持 force-loss 二阶导训练。

```bash
PYTHONPATH=python python -m model infer \
  --checkpoint ../deepmd-kit/examples/water/dpa4/lmp/pretrained.pt \
  --system ../deepmd-kit/examples/water/data/data_0 \
  --device cpu --operator-backend reference
```

实际模型 E/F/V 对比：`DPA4_RUN_DEEPMD_TESTS=1 PYTHONPATH=python python -m unittest tests.python.test_deepmd_integration -v`。当前本地默认 Python3.9/缺 DeePMD 编译库，该集成测试需在已有 DeepMD 环境运行；模块合同测试不代替它。

## 本地检查 Python 框架

需要已有 Python ≥3.9、PyTorch、NumPy；shape 和算子 reference 不依赖 DeePMD 导入。以下从工程根目录运行：

```bash
cd /Users/lipengfei/Desktop/work/bysj/myproject
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest discover -s tests/python -v

# 从 water 权重读取真实模型维度，输出逐阶段输入/输出 shape JSON
python3 -m model shape \
  --checkpoint ../deepmd-kit/examples/water/dpa4/lmp/pretrained.pt \
  --atoms 192 --edges 12288 --frames 1
python3 -m model operators
```

`--atoms` 是所有帧展平后的总原子数，`--edges` 是有向边容量；shape 命令用 meta 张量，不计算能量/力，不会载入权重执行模型。缺省 `DPA4Model(config)` 使用显式占位 backend，遇到未实现算子会抛 `NotImplementedError`，不伪造预测。

## DCU 构建与验证

在已经装好 **ROCm/DTK PyTorch** 的远程工程目录执行，使用同一个 Python 环境编译和运行；本项目不覆盖系统依赖、不需要升级 torch.compile 版本。代码走 eager extension 路径。

```bash
cd /path/to/myproject
source /opt/dtk/env.sh
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import torch; assert torch.version.hip and torch.cuda.is_available(); print(torch.__version__, torch.version.hip, torch.cuda.get_device_name())'
DPA4_BUILD_HIP=1 MAX_JOBS=4 python setup.py build_ext --inplace

# 必须先确认扩展已加载，否则 unittest 的 HIP 测试会 skip
python -c 'from dcu_dpa4.native import native_available; assert native_available(), "HIP extension not available"'
python -m unittest discover -s tests/csrc -v
python -m tests.csrc.benchmark --atoms 192 --edges 12288 --channels 16 --radial 6
```

若环境配置脚本不在 `/opt/dtk/env.sh`，改用集群已安装 DTK 的真实路径。架构设置使用当前 PyTorch/DTK 识别出的设备；跨节点部署时再按实际 `rocminfo` 配置 `PYTORCH_ROCM_ARCH`，不要套 NVIDIA 架构。

`CUDAExtension` 是 PyTorch 在 ROCm 环境中使用的 extension API 名称；`.cu` 源码由它 HIP 化并使用 HIP 编译器。`torch.cuda` 在 ROCm 中同样表示 DCU 设备接口。本项目的 kernel 使用 `hipLaunchKernelGGL`，不是要求 NVIDIA CUDA 运行时。

首批接口位于 `dcu_dpa4.ops`，统一 `backend="reference"` 或 `backend="hip"`。HIP 请求不能静默退回 CPU/Torch。目前仅支持连续 DCU FP32 数值输入，索引 INT64；Python wrapper 会做连续化。矩阵 weight 为 `[K,C]`，不是 `nn.Linear` 的 `[C,K]`。

## 原 DeePMD 模型作为 oracle

只在已装 DeePMD 的 Python ≥3.10 环境运行；不要把这段解释为项目独立模型已经完成。默认 operator_backend="deepmd"，设为 reference/hip 可使用新的实例级适配层。

```python
from model.runner import configure_device, DPA4Runner
from model.system import load_deepmd_system

configure_device("cpu")  # 在任何 deepmd backend 导入之前；DCU 时为 "dcu"
runner = DPA4Runner("../deepmd-kit/examples/water/dpa4/lmp/pretrained.pt")
system = load_deepmd_system("../deepmd-kit/examples/water/data/data_0")
prediction = runner.eval(system)
print(prediction.energy.shape, prediction.force.shape, prediction.virial.shape)
```

使用前确认数据目录存在（water 数据布局以当前 checkout 为准）。如果 DCU 的 vesin 报缺 `libcuda.so`，可以用 `neighbor_backend="cpu-bridge"` 作诊断：邻居搜索在 CPU、描述符在 DCU，**不是全 DCU 方案，也不应作为加速基线**。

## 完成边界

native 已写 geometry/radial/RMSNorm/linear/segment sum/force-virial，以及 fused edge-features（geometry + Bessel/C3 + 一阶 VJP）；独立 Python reference 路径已接入 OMol25 活跃分支的 SO(2)、attention、SO(3) grid、readout/fitting 组合与 checkpoint 映射。它们尚未在目标 checkpoint 完整加载或数值验证；SO(2)/Wigner/grid 等主体仍非 HIP native，另已加入 opt-in persistent HIP task worker，但当前只执行融合 `edge_features` task 或 `edge_geometry → radial_basis` 两阶段子图；完整 DPA4 DAG 尚不能交给 worker 执行。该 runtime 尚未编译或实机验证，HIP 正确性和性能结论必须由真实 DCU 验证。
