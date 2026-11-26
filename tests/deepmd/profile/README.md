# DPA4 热点分析：三种场景

从项目根目录执行下面命令。三种场景均测试原始 DeepMD：直接模式调用 `deepmd.infer.DeepPot`，由 DeepMD 自动选择 `pt` 或 `pt_expt` 等后端；LAMMPS 使用 DeepMD 导出模型及 C++ 后端。测试入口不导入本仓库模型适配器，也不加载本仓库 HIP 算子扩展。这里分析热点，不计算加速比；采集墙钟耗时包含 profiler 开销。

## 附带的水体系输入

已提供 [water_spce_4500.npz](water_spce_4500.npz)，由相邻 LAMMPS 仓库的 `examples/rdf-adf/data.spce` 转换而来。包含 **4,500 个原子／1,500 个水分子**，输入文件的类型顺序为 `O,H`（O=0，H=1）；直接推理会读取旁边 metadata 的 `type_map`，并按实际模型类型顺序重映射，因此也可用于包含 O/H 的多元素 OMol 模型。它是自动扩胞的基础输入，不限制最终 profiling 原子数。

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

扩胞生成的新 NPZ 会保留原类型编号，并将源输入的元素顺序写入 `type_map` 数组，因此生成文件单独复制后也可按模型顺序重映射。源输入没有元素信息时，可在生成或推理时传 `--input-type-map O,H`。

### 按原子数生成直接推理与 LAMMPS 共用的水体系

OOM 后可从原始 SPC/E data 中保留指定数量的完整水分子，同时生成 LAMMPS data 和直接推理 NPZ：

```bash
python3 tests/deepmd/profile/test_dpa4_inference_profile.py generate-data \
  --geometry ../lammps/examples/rdf-adf/data.spce --atoms 300 \
  --output-data /tmp/water_spce_300.data --output-npz /tmp/water_spce_300.npz
```

`--atoms` 必须是 3 的倍数，范围为 3 到原始数据的 4500 个原子；两个输出文件均不会覆盖已有文件。生成的 NPZ 内嵌 `O,H` 类型顺序，LAMMPS data 保留水分子的电荷、键、角及原周期盒。原盒不随删减缩小，因此小体系变得稀疏，适合功能与 OOM 排查，不适合与原体系直接比较性能。

直接推理使用 `--input-npz /tmp/water_spce_300.npz --fixed-input`，以避免重新自动扩胞；LAMMPS 命令使用 `--data-file /tmp/water_spce_300.data --replicate 1,1,1`。LAMMPS 仍须使用冻结后的 `.pt2` 模型和匹配的水模板。

下面的 LAMMPS 命令以相邻仓库 `../lammps` 为数据路径；`--lammps-exe` 仍需填写目标机器上编译出的可执行文件。

### 判断值不值得移植上游 fused SO(2) 卷积

```bash
python3 tests/dcu/fma_probe.py          # DCU：测 ffma 吞吐，折算成融合版的毫秒数
```

它把 `225.3 kFMA × E × layers × blocks` 折成时间，与稠密路径的 163.8 / 416.2 ms 比：
投影低于稠密才值得动上游那 24 个 `.cu`。见 [docs/execution-model.md](../../../docs/execution-model.md) §5。

### 本 checkout 的 DCU 路径

```bash
source tests/deepmd/profile/dcu_paths.env   # MODEL / NPZ / OUT
```

`MODEL` 是 DPA4-Air-OMol25-100M，`NPZ` 是 3000 原子 SPC/E 水盒（181838 条边），
与 `docs/op-time-share.md` 的 trace 同形；`OUT` 是验证产物目录。

## 1. 单卡 DeepMD 单次推理

```bash
source /opt/dtk/env.sh && python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
  --model-file /models/water.pt --input-npz tests/deepmd/profile/water_spce_4500.npz \
  --device cuda:0 --memory-fraction 0.70 --output-dir /root/private_data/lipengfei/test_result/profile-direct
```

- 默认根据实际可用显存自动周期扩胞，适用于单卡 64 GB；使用 70% 可用显存作为增量预算，为 profiler 保留余量。探测超预算或 OOM 时回退，只接受预算内的候选；这不是全局最大原子数的保证。
- 没有周期盒时保持输入规模。`--max-atoms N` 设置上限，基础输入已超限会报错。`--fixed-input` 禁用自动扩容。
- 选定规模后额外预热 5 次、同步，再采集一次完整 eval（包括能量、力、virial 和接口中的数据转换）。`--warmup` 可调整预热次数。
- `dpa4` 默认开启 profiler 和 Python stack capture，会把可采集的 `python_function` 调用路径及文件、行号写入 Chrome trace，供 Perfetto 展开查看。C++ 内部函数和设备 kernel 不会变成 Python 调用树。
- 编译推理默认关闭；仅对直接 `dpa4` 命令添加 `--compile`，即可在 `DeepPot` 初始化期间设置 `DP_COMPILE_INFER=1`。关闭时会显式设置为 `0`，并在模型构造后恢复原环境变量。编译可能在预热/扩容探测调用中产生额外耗时，这些调用不计入最终测时；比较延迟时建议使用同一模型、`--fixed-input` 和 `--no-profile`。DeepMD 文档目前注明该路径需要 `torch==2.11` / CUDA ≥ 12.6，而本脚本是 HIP/DCU runner，因此启用前需确认当前 DeepMD/设备组合支持；开关只请求编译路径，不保证加速；LAMMPS 子命令不受影响。
- 需要测不带 profiler 的推理时间时，在 `dpa4` 命令后添加 `--no-profile`。程序仍执行预热，然后同步测量 `DeepPot.eval`；默认测 10 次，可用 `--iterations N` 调整。输出 `dpa4_inference_summary.json`，报告均值、中位数、p95、最小值和最大值，不生成 Chrome trace。建议搭配 `--fixed-input` 做可重复比较。
- 始终使用原始 `DeepPot.eval`，支持当前 DeepMD 安装可读取的 `.pt/.pth/.pt2/.pte` 等模型；不要求 eager `pt_expt`，不需要编译本仓库扩展，也没有 `--native-ops` 参数。`--expected-dcu-name ''` 可关闭默认 K500SM_AI 名称检查。
- NPZ 包含 `coords`、`atom_types` 和可选 `cells`，只使用第一帧。geometry 支持显式标注 `Atoms # atomic/charge/molecular/full` 的正交 LAMMPS 数据，正确跳过 image flags。`--input-type-map O,H` 可显式指定输入编号对应的元素，优先于 NPZ 内嵌的 `type_map` 和旁边同名 `.metadata.json` 的 `type_map`（依次读取）；都没有时，按模型自身编号解释并检查范围。报告保存模型类型顺序和重映射表。

### 选择 Triton 推理路径（`--triton-infer` / `--triton-train`）

DPA4/SeZM 的 Triton 算子按 `DP_TRITON_INFER` 分级启用，而默认值 `0` 会走稠密参考实现；不显式设置时，profile 结果无法说明测的是哪条路径。

```bash
python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
  --model-file /models/water.pt --input-npz tests/deepmd/profile/water_spce_4500.npz \
  --fixed-input --no-profile --iterations 10 --triton-infer 2 \
  --device cuda:0 --expected-dcu-name '' --output-dir /root/private_data/lipengfei/test_result/profile-triton2
```

- `--triton-infer {0,1,2,3}` 与 `--triton-train {0,1}` 在构造模型之前设置对应环境变量，结束后恢复原值；不传则完全不干预环境，summary 的 `triton` 字段记录生效级别、来源（`cli`/`environment`/`unset`）以及 `TRITON_AVAILABLE`。
- `DP_TRITON_CONFIG_LOG=1` 让每次 launch 配置解析打印一行 `[dp-triton-config]`（含设备表名、来源与最终配置）。设备名没有内置表时会静默回落到保守默认，这一行是唯一能把"没调优"和"调优过"分开的证据。
- LAMMPS 子命令不接受这两个开关：它们只影响直接推理路径。

### 融合算子：`--fused` 与两次运行的对比

下面两组推理运行及其对比结果写到 `/root/private_data/lipengfei/test_result` 下（运行产物，不入库）；
数据生成、LAMMPS 与 trace 分析示例仍用 `/tmp`。

`dpa4` 子命令带 `--fused {off,on,softmax,envelope}`：在模型构造之后、warmup 之前调用
`python/model/fused_deepmd.py::deepmd_patch`，先把模块树与将要替换的两个接缝打印出来，再把
`segment_softmax` / `segment_sum` 绑进 DeepMD 的调用点。summary 的 `fused` 字段记录命中与回退计数
（`softmax_hits` / `softmax_fallbacks` / `envelope_hits` / `envelope_fallbacks` / `bound_*` / `last_error`），
没有可用扩展时整段为 `bound_*=false`、不会假装生效。

```bash
# 基线
python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
  --model-file "$MODEL" --input-npz "$NPZ3000" --fixed-input --output-dir /root/private_data/lipengfei/test_result/base

# 打上融合接缝
python3 tests/deepmd/profile/test_dpa4_inference_profile.py dpa4 \
  --model-file "$MODEL" --input-npz "$NPZ3000" --fixed-input --fused on --output-dir /root/private_data/lipengfei/test_result/fused

# 两份 summary 对比：设备总时间、launch 数、逐 kernel 增量
python3 tests/deepmd/profile/compare_fused_runs.py \
  --baseline /root/private_data/lipengfei/test_result/base/dpa4_*/dpa4_inference_profile_summary.json \
  --patched  /root/private_data/lipengfei/test_result/fused/dpa4_*/dpa4_inference_profile_summary.json \
  --json /root/private_data/lipengfei/test_result/fused_compare.json
```

`compare_fused_runs.py` 只读 summary、不跑推理，因此任何机器都能用；设备总时间取
`top_profiler_events` 里 `dpa4/...` 的那条注解，逐 kernel 增量取自 `trace_analysis`。
能量/力的数值一致性不在 summary 里，需跑 DCU 上的
`tests/csrc/accuracy/test_fused_deepmd_parity.py`。

### 辅助脚本：trace 归因、相邻算子链与作用域内占比

```bash
# 1) 把一次 trace 归因到 kernel 家族、roofline 等价量与 cast 分组
python3 tests/deepmd/profile/analyze_trace.py /tmp/profile/dpa4_*/dpa4_inference_trace.json \
  --json /tmp/lv2.json

# 2) 找可融合的相邻算子链（n-gram），并核对前向/反向分区与总时长
python3 tests/deepmd/profile/scan_fusion_patterns.py \
  --trace /path/to/dpa4_inference_trace.json --top 25 --max-len 5 \
  --json docs/fusion-patterns.json

# 3) 只看模型自己的时间：按 nn.Module 作用域统计前向 + 反向的算子占比
python3 tests/deepmd/profile/op_time_share.py \
  /path/to/dpa4_inference_trace.json --module 'nn.Module: ModelWrapper_0' \
  --top 30 --json /tmp/share.json
```

- `analyze_trace.py` 按 `triton` / `aten` / `gemm` / `other` 分类统计设备时间，并给每个 kernel 一个 roofline 等价量（访存型按等效字节、GEMM 按等效 GFLOP）。**如果 trace 里没有 triton kernel，脚本会明确提示：该次测的是稠密参考路径，不能据此下调优结论。** `cast_report` 按"输入 dtype + 维度"分组 cast/copy；注意 CPU op 的耗时包含等待设备，单次远高于同组的调用是同步点而非拷贝成本。
- `scan_fusion_patterns.py` 用 `cuda_runtime` 的 `correlation` 把每次 launch 对回 `cpu_op`，得到"算子名 + 形状"的 launch 序列；它先打印与 summary 的自动比对（总数、前向/反向分区），再给出按时间排序的签名、2–5 元相邻链与形状分类。链上的 `saving_ms` 是"每项 3 路流 × 最大 numel"的粗估，只用于排序。
- `fx_graph.py` **打印某个模块的 FX 图**（唯一的 FX 图脚本）：`--target model|so2_conv`，
  `--list-targets` 列出可打印对象；`--trace-device cpu|cuda` 决定在哪追踪；新增目标只需在
  `TARGETS` 里加一行。
- 两份已生成的图在 [`docs/artifacts/`](../../../docs/artifacts/)：`dpa4_air_cpu.fx.txt`（整模型）、
  `dpa4_air_so2_conv.fx.txt`（一个 `SO2Convolution`），都是纯文本、带形状。
- `ab_report.py` 读一组端到端 A/B 的收益：设备时间、相对基线的 delta、conv 自身耗时、以及路由是否真的开启；
  判据见 [docs/so2-conv-hip.md](../../../docs/so2-conv-hip.md) §6.4。
- `verify_fused.py` 把融合算子的验证阶梯串成一条命令（`host → build → acc → lat → par → e2e`），
  e2e 段自动判"接缝命中 / launch 数 / 设备时间"三条判据；`--dry-run` 只打印命令，可在没有 DCU 的机器上检查。
  各步的判据见 [docs/operators.md](../../../docs/operators.md) §4。

- `op_time_share.py` 回答"模型内部的时间分给哪些算子"：家族汇总、算子汇总、算子+形状站点，各带前向/反向拆分。作用域用 `nn.Module` 事件选，且必须用 **launch 时间**判定（整张图在模块调用里入队，设备要晚得多才排空；按设备时间戳过滤会把反向整段丢掉）。逐站点的融合归属见 [docs/op-time-share.md](../../../docs/op-time-share.md)。

### 确认 `.pt` 实际使用的推理后端

直接模式由 DeepMD 根据模型内容选择 `pt` 或 `pt_expt`，不能仅凭 `.pt` 后缀判断。profiling 脚本加载模型后会立即打印 `DeepMD backend` 和 `lower_input_kind`，即使后续推理在生成 summary 前失败也能看到。也可在同一容器及 Python 环境中单独加载模型检查；下面的命令不执行 `eval`：

```bash
source /opt/dtk/env.sh
DEVICE=cuda LOCAL_RANK=0 PYTHONPATH=python python3 - <<'PY'
import deepmd
from deepmd.infer import DeepPot

model_file = "/root/private_data/lipengfei/model/DPA4-OMol/DPA4-Air-OMol25-100M-v20260820.pt"
model = DeepPot(model_file, auto_batch_size=False)
backend = model.deep_eval
print("deepmd package:", deepmd.__file__, flush=True)
print("backend:", type(backend).__module__, flush=True)
print("lower_input_kind:", getattr(backend, "metadata", {}).get("lower_input_kind"), flush=True)
model.close()
PY
```

`backend` 为 `deepmd.pt.infer.deep_eval` 时使用旧 `pt` 路径；为 `deepmd.pt_expt.infer.deep_eval` 时使用 `pt_expt` 路径。`pt_expt` 的 `lower_input_kind=graph` 对应 `neighbor_graph_method` 构图配置，`nlist` 对应 `nlist_backend`；旧 `pt` 路径可能不提供这个元数据。上面的 `deepmd package` 用来核对当前进程导入的安装位置。成功完成 profiling 后，summary 的 `backend` 字段也会记录实际后端。若遇到 `Failed to load libcuda.so`，参见 [HIP/DCU 邻居搜索排查](TROUBLESHOOTING.md)。

## 2. 单卡 LAMMPS 稳态模拟

```bash
source /opt/dtk/env.sh && python3 tests/deepmd/profile/test_dpa4_inference_profile.py lammps-single \
  --model-file /models/water.pt2 --data-file ../lammps/examples/rdf-adf/data.spce \
  --lammps-input tests/deepmd/profile/dpa4_water_profile.template \
  --lammps-exe /opt/lammps/bin/lmp --gpu-ids 0 --world-size 1 \
  --replicate 2,2,2 --steps 20 --profile-skip 5 --profile-calls 10 \
  --output-dir /tmp/profile-single
```

## 3. 单机多卡 LAMMPS 稳态模拟

```bash
source /opt/dtk/env.sh && python3 tests/deepmd/profile/test_dpa4_inference_profile.py lammps \
  --model-file /models/water-with-comm.pt2 --data-file ../lammps/examples/rdf-adf/data.spce \
  --lammps-input tests/deepmd/profile/dpa4_water_profile.template \
  --lammps-exe /opt/lammps/bin/lmp --mpi-exe /opt/mpi/bin/mpirun \
  --gpu-ids 0,1,2,3 --world-size 4 --replicate 2,2,2 \
  --steps 20 --profile-skip 5 --profile-calls 10 \
  --output-dir /tmp/profile-multi
```

`world_size` 必须等于 GPU ID 数量，一个 rank 对应一张卡；省略时从 GPU 数量推导。`lammps-mpi4` 是固定四卡别名。GPU ID 指当前可见列表的下标，已有 HIP/CUDA mask 会保留映射；两种 mask 冲突时明确报错。单卡和多卡若需比较热点，应使用相同模型配置、全局原子数与复制倍数。

LAMMPS 默认运行 20 步，跳过前 5 次**后端模型调用**，采集随后 10 次；初始化力计算包含在跳过次数中。调用次数不是 MD 步数，窗口不会包含所有 LAMMPS 初始化。采集窗口内的邻居表更新、rank 内 PyTorch 活动以及可见的通信事件可在 trace 中检查；Kineto 并非整个 MPI 程序的系统级追踪器。

水模板的 `pair_coeff * * O H` 按元素名映射 data 中的类型，模型须包含 O 和 H，并配合匹配的 full-style data；使用固定种子生成 300 K 初速、0.25 fs、NVE 积分。保留原 SPC/E harmonic bond/angle 项并叠加 DeepMD pair，作为现有样例的 profiling 配置，不代表纯 DeepMD 水模型。输出 `trajectory.lammpstrj` 可用于检查原子确实移动。其他体系应提供匹配的 MD 模板和积分设置。

LAMMPS 扩胞使用 `--replicate`，不自动试跑最大体系；可先在单卡确定能运行的规模，然后固定该规模做单卡/多卡分析。当前工具仅支持单机多卡。

## DeepMD 后端构建要求

直接 Python profiling 可使用原始 DeepMD 安装，不需要以下 C++ 修改。LAMMPS `.pt2` 的有限窗口采集需重新构建包含以下 profiler 支持的 `deepmd-kit`，并让 LAMMPS 加载新库；这些修改只添加采集功能，不替换模型计算算子：

- `source/api_cc/include/PTExptProfiler.h`
- `source/api_cc/include/DeepPotPTExpt.h`
- `source/api_cc/src/DeepPotPTExpt.cc`

使用已有的 HIP/DTK、libtorch 与 MPI 构建配置，重新编译并安装 PTExpt backend。MPI 还要求模型含 with-comm artifact，并且 DeepMD 构建支持相应通信路径。旧版本仅在 TorchScript 后端支持 `DP_PROFILER`，不能替代本次 PTExpt 修改。

新增配置由 launcher 自动传入：`DP_PROFILER` 是 trace 文件名前缀，`DP_PROFILER_SKIP=5` 和 `DP_PROFILER_ACTIVE=10` 控制调用窗口。多卡前缀带 rank，后端再添加 `_gpuN.json`。每个 trace 同时写出 `.json.summary.json`，记录期望/实际采集次数、是否完整和异常状态。未设置 `DP_PROFILER` 时不启用采集；不要在同一推理线程叠加其他 profiler。

## 查看结果与验证

直接模式输出 `dpa4_inference_trace.json`、`dpa4_inference_profile_summary.json`；LAMMPS 输出每 rank 的 trace、窗口状态、rank 日志及 `profile_summary.json`。summary 提供 device kernel、CPU op、通信事件热点。Chrome trace 可用 Perfetto 查看。

直接模式报告 `model_implementation=deepmd.infer.DeepPot`、实际 `backend`、`operator_source=original DeepMD` 以及输入类型映射。通信识别基于事件名称；未观察到不代表没有通信。各事件可能重叠，耗时相加不是推理延迟；多卡报告分别保留，不相加成整体延迟。任一 rank 缺 trace、缺 GPU kernel 或窗口不完整都会判失败并保存原因。

主机测试与 CPU Kineto 生命周期检查：

```bash
python3 -m unittest discover -s tests/deepmd -p 'test*.py' -v
```

CPU 检查不证明 HIP、实际模型或 MPI 已通过。设备验收需运行上述三个命令，确认每个 rank 使用不同 GPU、窗口有 10 个模型范围、无非有限能量/力或 lost atoms，并检查轨迹移动。
