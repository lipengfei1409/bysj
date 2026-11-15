# HIP/DCU 推理报 `Failed to load libcuda.so`

## 排查结论与验证范围

针对 DPA4-Air-OMol25 `.pt` 在 DTK 环境运行直接 profiling 时出现的：

```text
RuntimeError: failed to compute neighbors: Failed to load libcuda.so.
```

日志和源码指向 **vesin 的 NVIDIA CUDA 邻居搜索被用于 HIP/DCU tensor**。
这里的 `libcuda.so` 是 NVIDIA Driver API，并非 DTK 的 HIP 动态库。
调整 `--memory-fraction` 不会解决这个加载失败；不要把 HIP 库软链接成
`libcuda.so`，也不要用 CUDA toolkit 的 stub 库充当驱动。

本次检查了当前项目和相邻 `deepmd-kit` 的源码，以及 vesin 官方源码。
本机为 macOS/CPU PyTorch（2.8.0，`torch.version.hip=None`），没有服务器模型和
DCU；尚未在服务器复现或验证修复。服务器实际导入的 DeepMD 路径、版本、模型后端
仍须通过下述命令确认，不能只根据 `.pt` 后缀判断。

## 源码证据

1. `test_dpa4_inference_profile.py::_run_dpa4` 明确要求
   `torch.cuda.is_available()` 且 `torch.version.hip is not None`。
   `_construct_deepmd_model` 直接调用原始
   `DeepPot(..., auto_batch_size=False)`，没有加载本项目适配器/原生扩展，也未指定邻居后端。
2. 类型映射打印后，`_prepare_direct_system` 在默认自动扩容模式下先通过
   `_memory_delta_for_inference` 做一次基础输入的 `model.eval`。
   当前失败发生在这个阶段，早于 profiler 上下文和正式采集。
   日志中的 O→7、H→0 与模型元素顺序一致。
3. 相邻 DeepMD 的
   `deepmd/pt_expt/utils/vesin_neighbor_list.py::is_vesin_torch_available`
   只检查 `import vesin.torch`，不验证 HIP 执行能力。
   `vesin_graph_builder.py::vesin_search_ijs` 将原设备上的
   `positions`、`box` 直接传入 `vesin.torch.NeighborList.compute`。
4. `deepmd/pt_expt/utils/graph_builder.py::resolve_auto_graph_builder`
   在 `device.type == 'cuda'` 且 nvalchemiops 可导入时选择 `nv`；否则，单帧且
   vesin 可导入时选择 `vesin`。HIP tensor 的设备类型同样是 `cuda`。
   nvalchemiops 的可用性检查也主要依赖导入成功，不能据此认定支持 HIP。
5. 旧 `pt` 路径的 `deepmd/pt/model/model/sezm_model.py::_select_neighbor_builder`
   同样会选择 nv 或 vesin。其 `use_self_built_nlist()` 返回 True，外层
   `DeepEval._setup_nlist_backend` 会忽略 `nlist_backend`。
6. [vesin 官方 C++ 绑定](https://github.com/Luthaf/vesin/blob/main/vesin-torch/src/vesin_torch.cpp)
   将 `device.is_cuda()` 映射为 `VesinCUDA`，并在原生调用失败时拼接
   `failed to compute neighbors:`，与日志一致。
   [其 CUDA 依赖 gpu-lite](https://github.com/metatensor/gpu-lite#cudahip-support)
   标注 CUDA 支持、HIP 计划中，运行时动态加载 NVIDIA CUDA 库。
   [PyTorch 官方说明](https://docs.pytorch.org/docs/stable/notes/hip.html)
   明确 HIP 复用 `torch.cuda` 和 `cuda` 设备名。

## 服务器最小验证：先脱离 DeepMD 和模型

在出错的同一容器、同一个 Python 环境执行。两个设备各启动一个进程，避免
一次原生崩溃影响另一个测试；输出保留设备名称和报错前标记。

```bash
source /opt/dtk/env.sh
for probe_device in cpu cuda:0; do
  PROBE_DEVICE="$probe_device" python3 -X faulthandler - <<'PY'
import importlib.metadata
import os
import torch

device = os.environ['PROBE_DEVICE']
print('torch:', torch.__version__, 'hip:', torch.version.hip,
      'cuda:', torch.version.cuda, 'test device:', device, flush=True)
for package in ('deepmd-kit', 'vesin', 'vesin-torch', 'nvalchemi-toolkit-ops'):
    try:
        print(package, importlib.metadata.version(package), flush=True)
    except importlib.metadata.PackageNotFoundError:
        print(package, 'not installed', flush=True)
points = torch.tensor([[0., 0., 0.], [1., 0., 0.]],
                      dtype=torch.float64, device=device)
print('torch tensor operation:', (points * points).sum().item(), flush=True)
if device != 'cpu':
    print('device name:', torch.cuda.get_device_name(points.device), flush=True)
import vesin.torch
print('vesin module:', vesin.torch.__file__, flush=True)
nl = vesin.torch.NeighborList(cutoff=2.0, full_list=True)
print('before vesin.compute', flush=True)
i, j, shifts = nl.compute(points=points,
                         box=10 * torch.eye(3, dtype=torch.float64, device=device),
                         periodic=True, quantities='ijS')
print('neighbors:', i.cpu().tolist(), j.cpu().tolist(), flush=True)
del nl
print('vesin object released', flush=True)
PY
done
```

预期鉴别结果：CPU 的两个有向邻居为 0→1、1→0；HIP 上普通 tensor 运算成功，
但 vesin.compute 再现 libcuda 报错，即可将问题隔离到 vesin 的设备实现。
若 CPU 也失败，先根据其具体错误检查 vesin/torch ABI 和包安装，不强行套用这个结论。

## 确认实际模型后端

```bash
DEVICE=cuda LOCAL_RANK=0 python3 -X faulthandler - <<'PY'
import sys
import deepmd
from deepmd.infer import DeepPot

print('python:', sys.executable, 'deepmd:', deepmd.__file__, flush=True)
model = DeepPot('/root/private_data/lipengfei/model/DPA4-OMol/'
                'DPA4-Air-OMol25-100M-v20260820.pt', auto_batch_size=False)
backend = model.deep_eval
print('backend:', type(backend).__module__, flush=True)
print('lower_input_kind:', getattr(backend, 'metadata', {}).get('lower_input_kind'),
      flush=True)
print('neighbor_graph_method:', getattr(backend, '_neighbor_graph_method', None),
      flush=True)
close = getattr(model, 'close', None)
if callable(close):
    close()
PY
```

## 按模型路径处理

| 实际路径 | 可行处理 | 注意事项 |
| --- | --- | --- |
| `pt_expt`，`lower_input_kind=graph` | 构造时传 `neighbor_graph_method='ase'`；已编译相应 DeepMD CPU op 时也可评估 `'cell'` | CPU 构图，图传回 HIP，模型仍在 DCU 执行。ASE 必须已安装。 |
| `pt_expt`，nlist 路径 | 构造时传 `nlist_backend='native'` | dense 邻居搜索可能耗用大量主机内存，先固定基础输入规模。 |
| 旧 `pt` SeZM 自建邻居路径 | 在 DeepMD 内部修复邻居后端选择/搜索，或用经验证的对应 graph 导出模型 | 外层 `nlist_backend='native'` 被忽略，不能解决其内部 vesin/nv 调用。 |

对于第一种路径，修改 profiling 脚本 `_construct_deepmd_model` 的这一处构造即可
进行一次手动验证（这是待在服务器验证的方案，本次未改运行代码）：

```python
return DeepPot(
    str(args.model_file),
    auto_batch_size=False,
    neighbor_graph_method='ase',
)
```

随后用原命令添加 `--fixed-input`，先验证 4,500 原子完整推理，再恢复自动扩胞。
当前 CLI 没有 `--neighbor-graph-method` / `--nlist-backend` 参数，不能直接在命令行
添加这两个选项。不要同时给 graph 模型传 `nlist_backend='native'`，当前 DeepMD 会拒绝。

长期兼容修复的落点是 DeepMD 共享的 `vesin_search_ijs`：在没有 HIP 实现时，
明确将 detached 坐标/盒子移到 CPU 搜索，仅把 `i,j,S` 返回原设备；后续用原始
坐标重建位移，保留力/virial 的梯度路径。两个自动选择器还应按设备执行能力筛选
`nv`，不能仅凭 `cuda` 设备名和包可导入就启用 NVIDIA 路径。
这种方案包含 CPU 搜索与传输，不能称为全程 GPU 邻居构建。

## Segmentation fault 如何判断

日志中 Python 已打印 `profile failed`，随后才收到 SIGSEGV。
UCX 的 `ucs_handle_error` 是信号处理栈；这个短栈不能证明 UCX 就是致因。
vesin 绑定的析构函数会调用 `vesin_free`，初始化失败后的释放和进程退出清理都
值得检查，但目前没有完整 native 栈，不能确定是哪一种。

先观察上述无 DeepMD/无模型的 vesin 最小实验是否同样崩溃：若同样崩溃，继续
围绕 vesin/其运行库取 native 栈；若只完整模型进程崩溃，再检查 DeepMD 引入的
其他库。需要时将最小实验保存为脚本，用 gdb 的 `run` 和
`thread apply all bt full` 获取实际故障帧。`faulthandler` 可补 Python 栈，不能
代替 gdb 的原生栈。不要以关闭 UCX 信号处理、屏蔽异常或 `os._exit()` 作为修复。

最终设备验收需确认：完整 eval 成功，能量/力有限，邻居边集与 CPU 参考一致，
并确认所选搜索路径与报告一致；若比较两种构图方案，还应核对能量/力数值一致性。
