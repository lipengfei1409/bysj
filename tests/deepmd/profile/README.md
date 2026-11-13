# DPA4 热点分析：三种场景

从项目根目录执行下面命令。直接模式使用本仓库 `DPA4Model`；LAMMPS 使用 DeepMD 导出模型及 C++ 后端，不会安装 Python native 补丁。这里分析热点，不计算加速比；采集墙钟耗时包含 profiler 开销。

## 附带的水体系输入

已提供 [water_spce_4500.npz](water_spce_4500.npz)，由相邻 LAMMPS 仓库的 `examples/rdf-adf/data.spce` 转换而来。包含 **4,500 个原子／1,500 个水分子**，模型类型顺序须为 `O,H`（O=0，H=1）。它是自动扩胞的基础输入，不限制最终 profiling 原子数。

| 数组 | 形状 | 类型 |
| --- | --- | --- |
| `coords` | `(1, 4500, 3)` | float64 |
| `atom_types` | `(1, 4500)` | int32 |
| `cells` | `(1, 9)` | float64 |

周期盒边长为 `35.50635 × 35.50635 × 35.44719 Å`。转换使用 full-style 的真实 xyz 列，忽略尾部 image flags；平移盒原点到零并周期折回，不改变周期几何。来源校验和、转换步骤及 O–H 距离检查记录在 [metadata](water_spce_4500.metadata.json)。NPZ 只保存模型需要的几何与类型，不含键角拓扑、电荷和速度；LAMMPS 仍直接读取原始 `data.spce`。

如需固定 36,000 原子的 NPZ，可从项目根目录生成：

```bash
python3 tests/deepmd/profile/test_dpa4_inference_profile.py generate-data \
  --input-npz tests/deepmd/profile/water_spce_4500.npz --replicate 2,2,2 \
  --output-npz /tmp/water_spce_36000.npz
```

下面的 LAMMPS 命令以相邻仓库 `../lammps` 为数据路径；`--lammps-exe` 仍需填写目标机器上编译出的可执行文件。

## 1. 单卡 DeepMD 单次推理

```bash
PYTHONPATH=python python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
  --model-file /models/water_pt_expt.pt --input-npz tests/deepmd/profile/water_spce_4500.npz \
  --device cuda:0 --memory-fraction 0.70 --output-dir /tmp/profile-direct
```

- 默认根据实际可用显存自动周期扩胞，适用于单卡 64 GB；使用 70% 可用显存作为增量预算，为 profiler 保留余量。探测超预算或 OOM 时回退，只接受预算内的候选；这不是全局最大原子数的保证。
- 没有周期盒时保持输入规模。`--max-atoms N` 设置上限，基础输入已超限会报错。`--fixed-input` 禁用自动扩容。
- 选定规模后额外预热 5 次、同步，再采集一次完整 eval（包括能量、力、virial 和接口中的数据转换）。`--warmup` 可调整预热次数。
- 默认启用 native GIE/RMSNorm，需 eager `pt_expt` `.pt`、HIP 扩展和受支持 DCU。`--native-ops ''` 选择 baseline，也允许导出模型；报告标注执行类型。`--expected-dcu-name ''` 可关闭默认 K500SM_AI 名称检查。
- NPZ 包含 `coords`、`atom_types` 和可选 `cells`，只使用第一帧。geometry 支持显式标注 `Atoms # atomic/charge/molecular/full` 的正交 LAMMPS 数据，正确跳过 image flags；类型下标必须匹配模型。

## 2. 单卡 LAMMPS 稳态模拟

```bash
python3 tests/deepmd/profile/test_dpa4_inference_profile.py lammps-single \
  --model-file /models/water.pt2 --data-file ../lammps/examples/rdf-adf/data.spce \
  --lammps-input tests/deepmd/profile/dpa4_water_profile.template \
  --lammps-exe /opt/lammps/bin/lmp --gpu-ids 0 --world-size 1 \
  --replicate 2,2,2 --steps 20 --profile-skip 5 --profile-calls 10 \
  --output-dir /tmp/profile-single
```

## 3. 单机多卡 LAMMPS 稳态模拟

```bash
python3 tests/deepmd/profile/test_dpa4_inference_profile.py lammps \
  --model-file /models/water-with-comm.pt2 --data-file ../lammps/examples/rdf-adf/data.spce \
  --lammps-input tests/deepmd/profile/dpa4_water_profile.template \
  --lammps-exe /opt/lammps/bin/lmp --mpi-exe /opt/mpi/bin/mpirun \
  --gpu-ids 0,1,2,3 --world-size 4 --replicate 2,2,2 \
  --steps 20 --profile-skip 5 --profile-calls 10 \
  --output-dir /tmp/profile-multi
```

`world_size` 必须等于 GPU ID 数量，一个 rank 对应一张卡；省略时从 GPU 数量推导。`lammps-mpi4` 是固定四卡别名。GPU ID 指当前可见列表的下标，已有 HIP/CUDA mask 会保留映射；两种 mask 冲突时明确报错。单卡和多卡若需比较热点，应使用相同模型配置、全局原子数与复制倍数。

LAMMPS 默认运行 20 步，跳过前 5 次**后端模型调用**，采集随后 10 次；初始化力计算包含在跳过次数中。调用次数不是 MD 步数，窗口不会包含所有 LAMMPS 初始化。采集窗口内的邻居表更新、rank 内 PyTorch 活动以及可见的通信事件可在 trace 中检查；Kineto 并非整个 MPI 程序的系统级追踪器。

水模板要求 O,H 类型顺序与匹配的 full-style data；使用固定种子生成 300 K 初速、0.25 fs、NVE 积分。保留原 SPC/E harmonic bond/angle 项并叠加 DeepMD pair，作为现有样例的 profiling 配置，不代表纯 DeepMD 水模型。输出 `trajectory.lammpstrj` 可用于检查原子确实移动。其他体系应提供匹配的 MD 模板和积分设置。

LAMMPS 扩胞使用 `--replicate`，不自动试跑最大体系；可先在单卡确定能运行的规模，然后固定该规模做单卡/多卡分析。当前工具仅支持单机多卡。

## DeepMD 后端构建要求

必须重新构建包含以下修改的 `deepmd-kit`，并让 LAMMPS 加载新构建的库：

- `source/api_cc/include/PTExptProfiler.h`
- `source/api_cc/include/DeepPotPTExpt.h`
- `source/api_cc/src/DeepPotPTExpt.cc`

使用已有的 HIP/DTK、libtorch 与 MPI 构建配置，重新编译并安装 PTExpt backend。MPI 还要求模型含 with-comm artifact，并且 DeepMD 构建支持相应通信路径。旧版本仅在 TorchScript 后端支持 `DP_PROFILER`，不能替代本次 PTExpt 修改。

新增配置由 launcher 自动传入：`DP_PROFILER` 是 trace 文件名前缀，`DP_PROFILER_SKIP=5` 和 `DP_PROFILER_ACTIVE=10` 控制调用窗口。多卡前缀带 rank，后端再添加 `_gpuN.json`。每个 trace 同时写出 `.json.summary.json`，记录期望/实际采集次数、是否完整和异常状态。未设置 `DP_PROFILER` 时不启用采集；不要在同一推理线程叠加其他 profiler。

## 查看结果与验证

直接模式输出 `dpa4_inference_trace.json`、`dpa4_inference_profile_summary.json`；LAMMPS 输出每 rank 的 trace、窗口状态、rank 日志及 `profile_summary.json`。summary 提供 device kernel、CPU op、通信事件热点。Chrome trace 可用 Perfetto 查看。

native 模块安装数量不能证明算子实际执行；直接模式根据 GPU kernel 名标记 `observed` 或 `unconfirmed`。通信识别基于事件名称；未观察到不代表没有通信。各事件可能重叠，耗时相加不是推理延迟；多卡报告分别保留，不相加成整体延迟。任一 rank 缺 trace、缺 GPU kernel 或窗口不完整都会判失败并保存原因。

主机测试与 CPU Kineto 生命周期检查：

```bash
python3 -m unittest discover -s tests/deepmd -p 'test*.py' -v
python3 tests/deepmd/profile/check_backend_profiler.py
PYTHONPATH=python python3 -m unittest discover -s tests/python -p test_dpa4_model.py -v
```

CPU 检查不证明 HIP、实际模型或 MPI 已通过。设备验收需运行上述三个命令，确认每个 rank 使用不同 GPU、窗口有 10 个模型范围、无非有限能量/力或 lost atoms，并检查轨迹移动。
