# SWE-Paddle Task Proposal: PaddlePaddle__Paddle-73854

## 1. 来源信息

- Instance ID: `PaddlePaddle__Paddle-73854`
- PR 链接: https://github.com/PaddlePaddle/Paddle/pull/73854
- PR 标题: `[0-size Tensor Job2 No.60] Add 0-size Tensor support for paddle.nn.functional.instance_norm`
- Base commit: `6ad407d1aaf34c41e193361d22f10c39946b715f`
- Gold commit: `81a61590467551723341382599b950b4e92a6e1e`
- Merged at: 2025-07-22
- 你的身份: contributor

## 2. 问题一句话

`paddle.nn.functional.instance_norm` 在动态图模式下对 0-size tensor（任意维度含有 0）的输入缺少显式处理，前向进入底层算子时出错，反向未正确填充梯度为 0，需要补齐 0-size tensor 支持。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该问题来自 Paddle 的「0-size Tensor 机制建设」系列任务，是真实研发需求，目标是为 `instance_norm` 算子补齐 0-size tensor 支持。
- **代表性**：覆盖 C++ phi kernel 层面的算子边界处理，涉及 forward kernel 的早期返回、backward kernel 的梯度填充和 InferMeta 的维度推断，是 Paddle 算子机制增强的典型样本。
- **边界清楚**：目标仅限 0-size tensor 输入的 forward/backward 早期返回逻辑；正向非零尺寸输入不应受影响。
- **非平凡性**：修复需要在 CPU kernel、GPU kernel、XPU kernel 和 InferMeta 多处分别添加 0-size 判断逻辑，涉及 `numel() == 0` 检查、输出分配、梯度填充（使用 `phi::Full` 或 `SetConstant`）和维度推断，不是简单机械修改。
- **回归护栏明确**：目标 F2P 可覆盖 0-size tensor 输入的 `instance_norm` 前向和反向调用；同文件已有的 `TestInstanceNormOp` 等标准算子测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[phi_kernel, instance_norm, 0-size_tensor, forward, backward, cpu, gpu, xpu]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：
  - `test/legacy_test/test_instance_norm_op_v2.py`（`TestInstanceNormOp_ZeroSize`）
- P2P 候选：同文件中已有的 `TestInstanceNormOp` 等标准 instance_norm 算子测试用例。
- 修复前预期：`base_commit` + `tests/test.patch` 后，0-size tensor 输入在 `instance_norm` 的前向/反向调用中失败（进入底层算子时出错）。
- 修复后预期：继续应用 `solution/code.patch` 后，0-size tensor 输入返回正确的空 tensor（shape 与预期一致），反向梯度填充为 0，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker: 无
- Dockerfile 或镜像地址: 暂无
- Paddle 来源: `PaddlePaddle/Paddle` source checkout at `base_commit`，需要源码编译。
- OS / Python / CUDA / cuDNN / 其他关键依赖: Linux CPU + Python + numpy；编译需要 CMake、GCC；不要求 CUDA/cuDNN（CPU 编译即可验证）。
- 硬件: CPU 即可（编译和测试均不需要 GPU）。
- patch 类型: C++ kernel 修改（`paddle/phi/kernels/cpu/instance_norm_kernel.cc`、`paddle/phi/kernels/gpu/instance_norm_kernel.cu`、`paddle/phi/kernels/xpu/instance_norm_kernel.cc` 等），需要重新编译。
- 最小测试命令: `bash tests/test.sh`
- 是否有 oracle 日志: 无

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述「instance_norm 对 0-size tensor 输入的行为异常」，不指出具体 `numel() == 0` 分支逻辑或具体代码位置。
- 环境风险：中。任务需要 Paddle 源码环境，patch 涉及 C++ 文件，需要重新编译。
- flaky 风险：低。测试使用固定的 0-size tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险：低。该 PR 目标集中在 instance_norm 的 forward/backward kernel 的 0-size 处理，测试明确指向零尺寸分支，适合作为一个独立样本。
- 其他不确定点：完整任务包阶段应确认新增 F2P（`TestInstanceNormOp_ZeroSize`）在 `base_commit` 上确实失败，并选择同文件中已有的 `TestInstanceNormOp` 等标准测试用例作为在 base 与修复后都稳定通过的 P2P nodeid。
