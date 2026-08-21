# 任务提案：PaddlePaddle__Paddle-56723

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-56723`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/56723
- PR 标题：`【complex op】No.7 add complex support for isclose`
- `base_commit`：待通过 merge commit 首个父节点确认（GitHub PR REST 返回的 `base.sha`，指向 PR 打开时 develop 分支的快照；可由 merge commit `d53972fd7fa1fab02859e95c5246cf9da19f0f03` 取 `git rev-parse d53972f^1` 得到）
- PR head：`9913ec22a72b60219af0b786372437e9e9b09dae`
- merged 时间：`2023-08-31T11:31:24Z`（merge commit `d53972fd7fa1fab02859e95c5246cf9da19f0f03`）
- 你的身份：原 PR 作者（GitHub @jinyouzhi）
- 后续联系人：GitHub @jinyouzhi

## 2. 问题一句话

`paddle.isclose` 算子不支持复数数据类型（complex64 / complex128），CPU/GPU kernel 注册中缺少复数类型列表、kernel 实现文件未特化复数 `IscloseFunctor` 与 CUDA kernel，且 Python 层 `check_variable_and_dtype` 与文档字符串未加入 complex64/complex128，导致传入复数 Tensor 时报 `TypeError` / kernel not found。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 Paddle 快乐开源「complex op」系列 No.7 的已合入 PR，属于 #56145「paddlepaddle 支持复数」的子任务，并经过 Paddle reviewer（@ScottWong98、@GGBond8488、@luotao1）审核。
- **代表性**：覆盖算子复数支持的完整垂直链路：C++ kernel 实现（CPU functor 特化 + GPU CUDA kernel 特化）、kernel 注册（CPU/GPU 两处注册列表补全 complex 类型）、Python API 层 dtype 检查与 docstring 更新，以及 legacy_test 单元测试新增；代表 Paddle 为现有算子补齐复数 dtype 支持时的典型分层修改模式。
- **边界清楚**：gold patch 只改动 5 个文件（2 个 kernel 注册 `.cc/.cu`、1 个 kernel 实现 `.h`、1 个 Python API `logic.py`、1 个测试文件），目标仅为 isclose 增加 complex64/complex128 支持；不引入新算子、不修改其他 math/logic OP，也不触碰 framework、distributed 等无关模块。
- **非平凡性**：模型需要同时在四层修改一致：① kernel_impl.h 为复数写 functor/CUDA 特化（含 NaN 处理、`abs()` 求复数模长）；② CPU/GPU 注册列表同时加入 complex；③ Python 两处 `check_variable_and_dtype` 与 docstring 同时加入 complex；④ 测试覆盖 static graph executor。只改其中任一层都会导致 dtype check 报错或 runtime kernel missing。
- **验收边界**：沿用原 PR 新增的 `TestIscloseOpCp64`、`TestIscloseOpCp128` 静态图用例以及继承自 `TestIscloseOp` 的 `TestIscloseOpComplex64`、`TestIscloseOpComplex128`；原始测试中已经存在的 float16/float32/float64 用例不纳入 F2P 但作为 P2P 回归基线。

## 4. 任务类型和标签

- 任务类型：`feature_implementation`
- 执行后端：`cpu_gpu`
- 设备范围：`cpu_and_gpu`
- 模块标签：`[ops, complex, kernel, phi, cpu_kernel, gpu_kernel, cuda, python_api, logic_op, legacy_test]`

## 5. 验证思路

- 目标测试文件：`test/legacy_test/test_isclose_op.py`
- 目标测试命令：

  ```bash
  python test/legacy_test/test_isclose_op.py -v TestIscloseOpComplex64 TestIscloseOpComplex128 TestIscloseOpCp64 TestIscloseOpCp128
  ```

  无 GPU 时最小可运行：

  ```bash
  python test/legacy_test/test_isclose_op.py -v TestIscloseOpComplex64 TestIscloseOpComplex128
  ```

- F2P 用例：
  - `TestIscloseOpCp64.test_cp64`（静态图 + complex64，仅在含 CUDA 环境下会走到 `CUDAPlace(0)` 分支；CPU 路径仍可创建 program 并验证不触发 dtype 报错）
  - `TestIscloseOpCp128.test_cp128`（静态图 + complex128）
  - `TestIscloseOpComplex64`（继承框架，覆盖 set_args + check_output 的 complex64）
  - `TestIscloseOpComplex128`（同上，complex128）
- 修复前预期：在 `base_commit` 上应用独立测试补丁后，Python 层因 `check_variable_and_dtype` 拒绝 complex dtype 而抛 `TypeError: The data type of input <...> is not supported`，或在极端跳过 dtype check 时因 kernel 未注册 complex 类型而抛 `Op (...) does not have kernel for data_type`。原 PR 未修改的 float16/float32/float64 P2P 用例继续通过。
- 修复后预期：继续应用 4 个非测试文件改动的 gold patch 后，目标 4 个 F2P 全部通过，static executor 输出 bool Tensor 与 NumPy `np.isclose` 结果一致；同时原有 float 系列 P2P 不受影响。
- P2P 候选：原文件中未被 PR 修改的 `TestIscloseOp`、`TestIscloseOpFp16`、`TestIscloseOpLargeDimInput`、`TestIscloseOpFloat64` 等 float 系列用例。
- 已完成的兼容性 Run/Test/Fix 预验证：建议在任务包阶段由 SWE-Paddle verifier 在 CPU + GPU 镜像内分别归档：①仅应用 test patch 时 F2P 应一致失败并输出 dtype/kernel 报错；②再叠加 gold patch 后 4 个 F2P 通过；③全部 P2P 通过。补丁需通过 `git diff --check`（无尾随空格、无新增断行错误）。

## 6. 环境与资源

- 是否能提供 Docker：有；建议优先使用带 CUDA 开发环境的官方 Paddle develop 镜像，以验证 `.cu` 编译与 GPU kernel。
- Dockerfile 或镜像地址：推荐 `paddlepaddle/paddle:latest-gpu-cuda11.8-cudnn8.6-trt8.5` 或更高版本，带 `nvcc`；CPU-only 验证可使用官方 CPU 镜像（仅覆盖 CPU kernel 与静态图路径）。
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`；测试补丁与 gold patch 来自 #56723 的 5 文件 PR diff（等价于 merge-base 到 head `9913ec2` 对 `isclose_kernel*`、`logic.py`、`test_isclose_op.py` 的改动）。
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：由于 gold patch 包含 C++/CUDA 源码级改动，**不适合使用预编译 wheel**；必须 source build 才能验证 kernel 注册与 functor 特化生效。
- OS / Python / CUDA / cuDNN / 其他关键依赖：推荐 Linux x86_64、Python 3.10、CUDA 11.8 + cuDNN 8.6、CMake ≥ 3.18、GCC ≥ 8.2、NumPy、`unittest`；CPU-only 验证可去掉 CUDA/cuDNN，但仍需编译 C++ CPU kernel。
- 硬件：GPU 测试需要至少 1 张支持 CUDA 的 NVIDIA 卡（验证 `.cu` kernel 注册）；CPU 路径仅需普通 x86_64 CPU。
- patch 类型：C++/CUDA（kernel 注册 + functor/CUDA 特化）+ 纯 Python（API dtype 检查与 docstring）+ 纯 Python legacy_test；**必须重新编译**才能让 kernel 改动生效。
- 最小测试命令：
  - CPU-only：`python test/legacy_test/test_isclose_op.py -v TestIscloseOpComplex64 TestIscloseOpComplex128`
  - 含 GPU：`python test/legacy_test/test_isclose_op.py -v TestIscloseOpCp64 TestIscloseOpCp128 TestIscloseOpComplex64 TestIscloseOpComplex128`
- 是否有 oracle 日志：完整任务包阶段由 SWE-Paddle verifier 分别在 base_commit（仅 test patch）和 gold patch 应用后归档正式 stdout/stderr 日志；重点记录 `TypeError`/`kernel not found` 的修复前报错栈与修复后 `OK` 结果。
- 兼容性说明：gold patch 中的 `phi::dtype::complex<float>` / `complex<double>` 类型需依赖 Paddle 已有 `paddle/phi/common/complex.h` 头文件与 `PD_REGISTER_KERNEL` 对 complex 的支持；这些基础设施在 PR base 时已存在。完整任务包阶段应优先 source build 精确基线并确认 `isclose` 在 base 上编译通过且 float 系列 P2P 通过，再依次叠加 test patch 与 gold patch。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述 `paddle.isclose` 在 complex64/complex128 输入下的可观察行为、static/dynamic 执行结果与 NumPy `np.isclose` 一致性要求，**不应**给出具体 kernel 文件名、functor 特化结构或 `check_variable_and_dtype` 修改点。
- 环境风险：任务需重新编译 Paddle C++/CUDA 部分，编译链版本（CUDA、GCC、CMake）不匹配会导致 `.cu` 编译失败；verifier 需固化镜像并把 `python setup.py build_ext` 或 `pip install -v -e .` 的完整日志纳入失败诊断。
- flaky 风险：中低。`TestIscloseOpCp64/Cp128` 使用 `np.random.rand` 生成输入但比较 `paddle.isclose` 与 `np.isclose`（默认 rtol/atol 固定），数值稳定性较高；唯一 flaky 点是 GPU 异步执行未显式同步，但 legacy_test 中 `exe.run` 已完成同步，风险可控。
- 拆分风险：PR 的 5 个文件改动属于「复数支持」同一语义单元——缺少 kernel 注册会 runtime 失败，缺少 functor 特化会编译失败，缺少 Python dtype 检查会在 API 层被拦截，缺少测试无法验证行为；按 PR 粒度保留为一个样本最自然。
- 依赖风险：任务依赖 Paddle 已引入的 `phi::dtype::complex<T>` 类型与 `paddle/phi/common/complex.h`（由 #56145 系列前序 PR 提供），这些基础能力不重复纳入本任务 gold patch；patch 提取时需确认 base 上 `#include "paddle/phi/common/complex.h"` 能找到。
- patch 提取风险：GitHub 记录的 PR 是从 fork 分支 `jinyouzhi:isclose_complex` 合并入 develop，8 个 commit 中前两个为功能/测试，后 6 个为 reviewer 反馈和编译修复（模板编译问题、CUDA 编译错误、abs 错误处理、dtype 列表补齐等）。**gold patch 应使用 squash 后的 5 文件合并 diff** 而非逐 commit 应用，避免中间状态无法编译或测试失败。提取时需从 GitHub PR Files 页导出的 5 文件 unified diff（或 `9913ec2` 与 merge-base 的差异）为准，并验证能干净应用到 `base_commit`、一次编译通过、P2P 不回归、F2P 通过。
