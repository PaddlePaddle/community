# SWE-Paddle Task Proposal: PaddlePaddle__Paddle-73698

## 1. 来源信息

- Instance ID: `PaddlePaddle__Paddle-73698`
- PR 链接: https://github.com/PaddlePaddle/Paddle/pull/73698
- PR 标题: `[0-size Tensor No.160、162、164] Add 0-size Tensor support for conv1d_transpose`
- Base commit: `3ad243cee3fc076300af25b9c806c47a09d7aa5c`
- Gold commit: `cd827ae7ab4552dfb563b070c569d1bf919bba78`
- Merged at: 2025-06-30
- 你的身份: contributor

## 2. 问题一句话

`paddle.nn.functional.conv1d_transpose/conv2d_transpose/conv3d_transpose` 在输入为 0-size Tensor 时，CPU/GPU/XPU kernel 未处理 0-size 边界情况导致崩溃或报错，需要在 kernel 入口添加 0-size 早期返回逻辑，并修复 InferMeta 的形状计算。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**: 来自 Paddle「0-size Tensor 机制建设」系列任务，是真实研发需求。
- **代表性**: 覆盖 C++ kernel 层面的 0-size Tensor 边界处理，涉及 CPU/GPU/XPU 三端 kernel、梯度 kernel 和 InferMeta。
- **边界清楚**: 目标仅限输入为 0-size 时的 kernel 早期返回和形状计算；正向非零尺寸输入不应受影响。
- **非平凡性**: 修复需要在多个 kernel 中添加 `input.numel() == 0` 的早期返回，并使用 `phi::Full` 填充全零，同时修复 InferMeta 对 0-size 输入的形状推导和 output_size 检查，涉及对 kernel 执行流程和形状推导机制的理解。
- **回归护栏明确**: 目标 F2P 可覆盖 0-size Tensor 输入的 `conv1d_transpose/conv2d_transpose/conv3d_transpose` 算子测试；同文件中已有的标准测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型: `bug_fix`
- 执行后端: `cpu`
- 设备范围: `cpu_only`
- 模块标签: `[operator_kernel, conv1d_transpose, conv2d_transpose, conv3d_transpose, 0-size_tensor, cpu_kernel, gpu_kernel, xpu_kernel]`

## 5. 验证思路

- 目标测试命令: `bash tests/test.sh`
- 目标测试文件:
  - `test/legacy_test/test_conv2d_transpose_op.py`（`TestFunctionalConv2DTranspose_ZeroSize`、`TestFunctionalConv2DTranspose_ZeroSize2`）
  - `test/legacy_test/test_functional_conv1d_transpose.py`（`TestFunctionalConv1DTranspose_ZeroSize`、`TestFunctionalConv1DTranspose_ZeroSize2`）
  - `test/legacy_test/test_functional_conv3d_transpose.py`（`TestFunctionalConv3DTranspose_ZeroSize`、`TestFunctionalConv3DTranspose_ZeroSize2`）
- P2P 候选: 同文件中已有的 `TestConv2DTransposeOp`、`TestFunctionalConv1DTranspose`、`TestFunctionalConv3DTranspose` 等标准算子测试用例。
- 修复前预期: `base_commit` + `tests/test.patch` 后，0-size Tensor 输入的算子测试失败（kernel 崩溃或报错）。
- 修复后预期: 继续应用 `solution/code.patch` 并重新编译后，0-size Tensor 输入正常返回全零 Tensor，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker: 无
- Dockerfile 或镜像地址: 暂无
- Paddle 来源: `PaddlePaddle/Paddle` source checkout at `base_commit`，需要源码编译。
- OS / Python / CUDA / cuDNN / 其他关键依赖: Linux CPU + Python + numpy；编译需要 CMake、GCC；不要求 CUDA/cuDNN（CPU 编译即可验证）。
- 硬件: CPU 即可（编译和测试均不需要 GPU）。
- patch 类型: 含 C++ kernel 修改（CPU/GPU/XPU 三端）+ InferMeta，需要重新编译 Paddle。
- 最小测试命令: `bash tests/test.sh`
- 是否有 oracle 日志: 无

## 7. 风险自查

- 泄露风险: 正式 `instruction.md` 只描述「conv1d_transpose/conv2d_transpose/conv3d_transpose 对 0-size Tensor 输入的行为异常」，不指出具体 `input.numel() == 0` 分支逻辑或具体代码位置。
- 环境风险: 中。任务涉及 C++ kernel 修改，需要源码编译 Paddle，编译时间较长。
- flaky 风险: 低。测试使用固定的 0-size Tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险: 低。该 PR 目标集中在 `conv1d_transpose/conv2d_transpose/conv3d_transpose` 的 CPU/GPU/XPU kernel 0-size 早期返回和 InferMeta 形状修复，测试明确指向新增的 ZeroSize 测试类，适合作为一个独立样本。
- 其他不确定点: 完整任务包阶段应确认新增 F2P 在 `base_commit` 编译后确实失败。注意该 PR 同时修改了前向和反向 kernel，以及 InferMeta。
