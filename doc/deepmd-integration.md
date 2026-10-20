# 接入 DeePMD-kit 的 Python 适配层

## 接入方式

推荐入口是 `dcu_dpa4.deepmd`，不是独立 shape skeleton。它在 DeePMD 原模型的**实例**上替换部分 forward，继续使用原模型所有权重、type map、邻居图、Wigner/SO2/grid/FFN/fitting、自动求导和 force/virial 组装。没有修改相邻 deepmd-kit 源码，没有全局覆盖类，没有重新注册 `dpa4` 类型。

Python 包导入不会自动安装 adapters，也不会导入 DeepPot、设置环境变量或修改 sys.path。安装 handle 可查询调用计数并恢复，既可做逐阶段数值对照，也可在后续扩充映射到新的 native operators。

## 已有 DeepPot：最少改动

```python
from dcu_dpa4.deepmd import configure_eager, install_operators

# 必须在构建模型前设置；改变 DEVICE 应在新进程中进行。
configure_eager("cpu")  # ROCm 环境改为 "dcu"
from deepmd.infer import DeepPot

dp = DeepPot("pretrained.pt", no_jit=True)
model = dp.get_model()  # 官方 Python 入口，不访问 dp.deep_eval.dp.model 私有层级

# coord/cell/atype 完全沿用原 DeepPot 数据格式。
with install_operators(model, backend="reference") as handle:
    energy, force, virial = dp.eval(coord, cell, atype)
    print(handle.report())  # module path / class / operator / backend / calls

# 退出上下文后 dp 已恢复原 forward；还可以继续 dp.eval。
dp.close()
```

`backend="reference"` 在原模块中执行项目的 Torch 计算；`backend="hip"` 使用编译好的 `_C`。不要在模型加载后才设置 fused flags：SO2 等模块会在构造期缓存加速选择。若模型已经使用 fused/compile，installer 会拒绝并要求先 configure_eager 再重新加载。

## checkpoint：使用 runner

```python
from model.runner import DPA4Runner
from model.system import load_deepmd_system

system = load_deepmd_system("../deepmd-kit/examples/water/data/data_0", max_frames=1)
with DPA4Runner(
    "../deepmd-kit/examples/water/dpa4/lmp/pretrained.pt",
    device="cpu", operator_backend="reference",
) as runner:
    prediction = runner.eval(system)
    print(prediction.energy.shape, prediction.force.shape, prediction.virial.shape)
    print(runner.operator_report)
```

runner `operator_backend` 默认为 `deepmd`（原算子），也可选 reference 或 hip。`runner.potential` 是原 DeepPot，`runner.torch_model` 是原 PyTorch 模型。`runner.eval(system, **eval_kwargs)` 把 fparam/aparam/charge_spin 等参数交给 DeepPot，模型实际是否接受仍取决于 checkpoint；不在适配层重写这些条件分支。

如果需要 lower-level 接口、extended atoms、ghost、已有 edge graph 或 comm_dict，可直接使用 `runner.torch_model.forward_lower(...)`，保持当前 DeePMD 的完整参数合同。实例级 primitive 适配仍生效；**不要把项目 shape graph 当作 LAMMPS ghost/scatter graph**。

## 支持的模块与兼容性

| DeePMD/Torch 原模块 | adapter | 保留内容 |
|---|---|---|
| `sezm_nn.radial.RadialBasis` | radial | 原 adam_freqs、rcut、exponent；reference 支持 Bessel/Gaussian，HIP 仅 Bessel |
| `sezm_nn.norm.RMSNorm` | rmsnorm | 原 adam_scale、eps、module dtype 与输出 dtype |
| `network.mlp.MLPLayer` | linear | 原 matrix/bias/activation/idt/residual、strict dtype promotion |
| `sezm_nn.so3.ChannelLinear` | linear | 原 `[Cin,Cout]` weight 与 bias、任意前导 batch 轴 |
| `torch.nn.Linear` | linear | 原 `[Cout,Cin]` weight（调用 kernel 时转置）与输出布局 |

匹配 exact class；`EquivariantRMSNorm`、`ReducedEquivariantRMSNorm`、`FocusLinear`、`SO3Linear` 不会误替换成普通 norm/linear。父模块已融合的路径不会被隐式拆开；未支持的模块继续用 DeePMD 原计算。`report()` 的 calls 可发现匹配模块是否实际被绕过。

- 模块类型、Parameter 对象、state_dict keys、module hooks 和 root model 的调用签名不变。reference radial 使用原 pi/gaussian 常量和 envelope 子模块；HIP radial 融合掉内部 envelope，因此它的子模块 hook 不再单独执行。
- 单张量 forward 的位置参数、关键字参数与 inspect.signature 保持一致；多维/非连续输入在 primitive 边界整理布局。
- reference 在线性 autocast 路径保留 Torch 原 F.linear 的 dtype 行为。HIP 仅连续化后的 DCU FP32，不自动 CPU fallback、dtype 降精度或 Gaussian fallback；HIP inference 禁止外部 autocast。
- HIP 仅支持 eval 模式的一阶 E/F/V 推理；调用需要 grad 的力推理不能套全局 inference_mode。当前 native radial/RMSNorm 的二阶导不完整，不能用于 force-loss 训练。
- 全局训练、PT2/TorchScript、torch.compile、freeze/export 不在本轮支持范围。卸载 adapters 后再序列化整个 Python 对象、deepcopy 或导出；正常 state_dict 的参数/命名保持不变。
- 这是 Python eager 的接入，不会自动让 `dp test`、LAMMPS C++ 插件或已导出的模型使用本项目 kernels；这些部署入口需要后续单独接入。
- 安装/卸载不得与同一模型执行并发；同一实例不能重复安装。不要在 adapters 激活期间由别的库再次覆盖它的 forward。

## 安装与运行

完整集成要求已有 DeePMD-kit 的 DPA4 环境和 Python ≥3.10（本项目 shape/reference primitives 仍可在3.9运行）。使用当前开发分支的 DeePMD，不要为了安装本项目替换它的系统依赖。

```bash
cd /path/to/myproject
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

# CPU / Torch reference 嵌入原模型
python -m model infer \
  --checkpoint ../deepmd-kit/examples/water/dpa4/lmp/pretrained.pt \
  --system ../deepmd-kit/examples/water/data/data_0 \
  --device cpu --operator-backend reference --max-frames 1

# ROCm/DTK：先按根 README 构建扩展，再运行
source /opt/dtk/env.sh
DPA4_BUILD_HIP=1 MAX_JOBS=4 python setup.py build_ext --inplace
python -m model infer \
  --checkpoint ../deepmd-kit/examples/water/dpa4/lmp/pretrained.pt \
  --system ../deepmd-kit/examples/water/data/data_0 \
  --device dcu --operator-backend hip --max-frames 1
```

若远程目录位置不同，使用 checkpoint/system 的实际绝对路径。若 vesin 仍依赖 libcuda.so，可显式加 `--neighbor-backend cpu-bridge` 诊断；这样邻居搜索仍在 CPU，并非全 DCU，也不宜作为性能基线。

## 验证

```bash
# 无需完整 DeepMD 的单元/模块合同测试
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python -m unittest discover -s tests -v

# 原 DeepMD 环境：实际加载 pretrained、比较安装前/后的 E/F/V 与恢复结果
DPA4_RUN_DEEPMD_TESTS=1 PYTHONPATH=python \
  python -m unittest tests.python.test_deepmd_integration -v

# DCU 原环境，另测试 HIP 在完整模型里的 forward 和坐标梯度
DPA4_RUN_DEEPMD_TESTS=1 DPA4_TEST_DEVICE=dcu \
  DPA4_TEST_OPERATOR_BACKEND=hip PYTHONPATH=python \
  python -m unittest tests.python.test_deepmd_integration -v
```

还可设置 `DPA4_TEST_CHECKPOINT`、`DPA4_TEST_SYSTEM`、`DPA4_TEST_NEIGHBOR_BACKEND`。默认先用 water 模型，不以文件名推测 Neo 的结构。

本机测试中的 upstream contract probes 读取本地真实 forward 函数，但隔离了构造和 compiled dependencies，**不是完整 DeepPot 数值测试**。开启 DPA4_RUN_DEEPMD_TESTS 后，缺依赖会直接失败，不会静默跳过。完整 E/F/V 与 HIP 实机验证仍需在已有 DeePMD/DCU 环境运行。

## 后续算子接入位置

在 `python/dcu_dpa4/deepmd/adapters.py::ADAPTERS` 为新的 exact upstream class 增加小型 forward adapter，并给出本地上游函数的合同测试与真实模型 E/F/V 对比。若新算子是多张量输入（例如 SO2/Wigner），先扩展 installer 的签名路由及 VJP 测试，不能直接套本轮单输入 wrapper。graph/megakernel 融合放在更大模块边界接入，仍由原 DeePMD 提供权重与 MD 数据流。
