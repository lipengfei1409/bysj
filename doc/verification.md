# 本轮验证记录

日期：2026-10-03。工作目录：`/Users/lipengfei/Desktop/work/bysj/myproject`。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python3 -m unittest discover -s tests -v
```

结果：38 项测试，30 项通过，8 项 HIP 测试因本机没有 DCU/HIP 扩展跳过。不能把这解释成 native 精度通过。

water pretrained metadata→DPA4Config→meta modeling 实测输出33个主阶段；N=192、E=12288、F=1时输出 energy `[1,1]`、atom_energy `[192,1]`、force `[192,3]`、virial `[1,3,3]`。只验证配置/shape 与接线，不执行 checkpoint 数值模型。

32个 Python 文件 AST 解析通过（随后增加的测试也由 unittest 解析运行）；45项算子合同均出现在数据流文档；源码无行末空白。`setup.py --name` 与 benchmark `--help` 正常。

使用本机 clang++ C++17、PyTorch/Python 头文件对 `csrc/bindings.cpp` 执行 `-fsyntax-only`，exit 0。这只覆盖 pybind/header 声明，不覆盖 HIP 翻译单元、链接、设备执行或精度。

远程 SSH 预检在 DNS 解析阶段失败（无法解析 ksai.scnet.cn）；没有修改远程工程或依赖。HIP 编译与8项 native 精度/梯度/stream 测试、性能基线仍需在远程实际 DTK 环境执行 README 命令。

## Python 接入适配层：增量验证

新增 `dcu_dpa4.deepmd` 实例级 adapters、runner/backend 选择和 infer CLI 后，重新运行同一全量测试命令：56项测试，47项通过，9项跳过（8项HIP、1项显式 opt-in 实际 DeePMD 集成）。`discover -s tests/python` 单独运行也通过：48项、1项集成测试跳过。

验证覆盖：真实 Torch Linear 的前向/参数梯度、非连续多维输入、空batch/零输入宽度、autocast、实例隔离、Parameter identity/state_dict、hooks/调用签名与异常恢复；读取本地真实 upstream forward 函数作 radial/RMSNorm/MLPLayer/ChannelLinear 合同探针，覆盖 stored constants 和 strict dtype promotion。这些探针隔离了模块构造/compiled dependencies，不是完整 DeepPot 数值验证。

当前39个 Python 文件全部AST解析通过，setup能发现 `dcu_dpa4.deepmd` 包；导入 integration 没有导入顶层 DeepMD、没有修改环境变量；infer CLI帮助正常。

实际 opt-in DeePMD 测试已尝试，当前本地 Python3.9在上游 `str | None` 注解阶段失败（该分支要求 Python ≥3.10）；先前模块导入预检还显示没有 `deepmd.lib`。现已给runner增加明确Python版本提示。没有为通过测试而覆盖上游依赖或伪装完整模型；实际 E/F/V与HIP验证待现有远程环境。

## 指令层：2026-10-09 增量验证

添加 `include/common`、thread/group 基元和 DUMMA 封装后，全量测试重新执行：61 项、51 通过、10 跳过（8 个 native HIP、1 个新 HIP 指令实机测试、1 个实际 DeePMD 集成）。

4 项指令层 host 测试通过：10 个头文件独立 C++17 `-Wall -Wextra -Werror` 语法编译；matrix view/尾块补零/边界 store/非零 C/reference MMA/D-C 别名/FMA 数值；非法 MMA shape 编译拒绝；host 显式开启 DUMMA 时拒绝而不是 fallback。初始实现前 host 合同测试因 API 缺失失败，随后通过。

sdist 在 `/tmp` 的隔离源码副本构建成功，检查归档确认包含10个头文件、native源码和HIP测试C++文件。直接在新的 bysj 工作区运行 sdist 时临时打包目录创建被沙箱拒绝，因此将打包验证转到可写的 `/tmp`，没有为此修改工作区权限。setuptools 报 author/url 元数据 warning，不影响本轮打包。

本机没有 hipcc/DCU。`du_mma.h` 的实际编译兼容性、gfx设备支持、LDS/fragment精度、生成ISA与性能全部未验证；硬件测试已提供在 `tests/csrc/instructions_hip.cpp`，只在真实 DTK 环境显式启用。没有替换 linear，也没有宣称完整 DPA4 加速。

### 指定 DUMMA 手册后的独立 API 封装

按手册第 2 节新增 `ops/group/dumma.cuh` 的四类独立 API，原固定 shape 类复用该层。先添加直接入口的 host 启用拒绝测试，确认因该头文件不存在而失败，再添加实现，测试通过。

重新运行全量回归：62 项、52 通过、10 跳过；5 项指令层 host 测试通过，现有11个头文件分别独立 C++17 编译通过。新增实机 direct/shape 两条数值路径，但本机仍未编译或执行其中的 DUMMA 分支。host 的成功只证明条件关闭/错误诊断和可移植部分，不证明 DTK native API 的编译、精度或性能。

## rocBLAS 封装：2026-10-09 增量验证

本地DTK rocBLAS手册5.3/5.6/7.2对应的host封装、Torch绑定与Python接口已编写。首轮Python接口/host布局测试因模块/头文件不存在失败；FP32系数下溢语义也单独经历失败→实现→通过。

最终全量回归78项、64通过、14跳过；新增9项Python BLAS合同与3项host布局/构建检查通过。覆盖非方形与4种transpose、batch、非连续输入、一阶梯度、beta=0不读NaN C、alpha=0、K=0/空batch、FP32 scalar舍入/下溢、非法dtype/shape、AMP拒绝、显式backend不fallback以及无HIP时拒绝开启BLAS构建。

本机clang++对`csrc/bindings.cpp`分别在未开启BLAS和`-DDPA4_ENABLE_ROCBLAS=1`时执行C++17语法检查，均exit0；这不涵盖实际rocBLAS header、`blas.cu`、链接或设备执行。benchmark帮助和默认`setup.py --name`正常。sdist在`/tmp`副本构建成功，归档确认有BLAS两个hpp、HIP源码、Python接口、native测试/基准与文档。

4项新增rocBLAS实机测试未执行，属于14项skip的一部分。本机没有hipcc/DCU，不能声称rocBLAS编译、矩阵精度、多stream/多卡隔离或性能通过；执行`doc/blas.md`中的构建及显式实机验证命令才能确认。没有自动替换原linear/DeePMD adapter，也没有声称实现hipBLASLt或device侧库调用。

## 独立 numerical modeling：2026-10-11 增量验证

新增 NumericalBackend、参数化 RadialMLP/TypeEmbedding/RadialBasis、prepare_edges 与严格部分 checkpoint loader。首轮缺失接口测试失败后实现；本轮最终命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python python3 -m unittest discover -s tests -q
```

结果：95 项，79 通过，16 跳过，exit 0。新增15项本地 numerical modeling 测试覆盖 stage 注册与输出合同、单一参数所有权、空边/mask/周期/多帧/非法拓扑、FP64、gradcheck、真实 water 前端参数、upstream radial/norm/linear forward 合同、GELU tanh 近似和 version state_dict roundtrip。测试专用数值图在 no_grad 下的力与有限差分一致，但该图不代表 pretrained DPA4。

独立代码审查发现四项重要问题，均先写回归、确认失败后修复：native geometry sibling 输出导致边向量 VJP 断路、native force 汇总不接受带梯度向量、GELU 近似不符、version 控制语义未序列化；审查复核15项测试通过，无剩余 Critical/Important。另对齐 safe_norm 的 eps 与 canonical +z padding。两个 native 边界探针只模拟 Python 拓扑/输入合同，不等于实机执行。

文档小系统示例实测：water version=1.1，加载4个前端权重张量，64个张量 deferred；rbf `[6,6]`，radial/fused `[6,2,16]`，全部前端输出有限，complete_model=False。完整独立 forward 仍在未实现 Wigner 处明确失败；未验证 water/Neo 完整 E/F/V。

新增两项 opt-in DCU 前端 reference 对比/坐标一阶梯度测试，当前均 skip。16项skip还包括原8项HIP、4项rocBLAS、1项DUMMA、1项实际DeePMD集成。本机无DCU，native编译/精度/性能仍待远程执行 `doc/numerical-modeling.md` 命令。本轮未提交Git或修改系统依赖。
