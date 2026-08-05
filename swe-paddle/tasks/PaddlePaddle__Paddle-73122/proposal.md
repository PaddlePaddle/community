# SWE-Paddle Task Proposal: PaddlePaddle__Paddle-73122

## 1. 来源信息

- Instance ID: `PaddlePaddle__Paddle-73122`
- PR 链接: https://github.com/PaddlePaddle/Paddle/pull/73122
- PR 标题: `[0-size Tensor No.112] Add 0-size Tensor support for multi_dot`
- Base commit: `2624aee95b82873848e34fc3e5673a1ac42f84c4`
- Gold commit: `29b711ba8db211ed31b3354562f62fb5ce568b40`
- Merged at: 2025-06-11
- 你的身份: contributor

## 2. 问题一句话

`paddle.linalg.multi_dot` 在输入中包含 0-size Tensor 时，C++ kernel 未处理 0-size 边界情况导致崩溃或报错，需要在 kernel 入口添加 0-size 早期返回逻辑，并对输出 numel > 0 的情况填充全零。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**: 来自 Paddle「0-size Tensor 机制建设」系列任务，是真实研发需求。
- **代表性**: 覆盖 C++ kernel 层面的 0-size Tensor 边界处理，涉及前向和反向两个 kernel。
- **边界清楚**: 目标仅限输入包含 0-size 时的 kernel 早期返回；正向非零尺寸输入不应受影响。
- **非平凡性**: 修复需要在 `MultiDotKernel` 和 `MultiDotGradKernel` 中分别添加 0-size 检查，且需要处理输出 numel > 0 的特殊情况（用 `phi::Full` 填充全零），涉及对多矩阵链式乘法语义和 kernel 执行流程的理解。
- **回归护栏明确**: 目标 F2P 可覆盖 0-size Tensor 输入的 `multi_dot` 算子测试；同文件中已有的 `TestMultiDotOp` 等标准测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型: `bug_fix`
- 执行后端: `cpu`
- 设备范围: `cpu_only`
- 模块标签: `[operator_kernel, multi_dot, 0-size_tensor, cpu_kernel, gpu_kernel]`

## 5. 验证思路

- 目标测试命令: `bash tests/test.sh`
- 目标测试文件:
  - `test/legacy_test/test_multi_dot_op.py`（`TestMultiDotOp_ZeroSize1`）
- P2P 候选: 同文件中已有的 `TestMultiDotOp`、`TestMultiDotOp_Float16` 等标准 multi_dot 算子测试用例。
- 修复前预期: `base_commit` + `tests/test.patch` 后，0-size Tensor 输入的 `multi_dot` 算子测试失败（kernel 崩溃或 BLAS 报错）。
- 修复后预期: 继续应用 `solution/code.patch` 并重新编译后，0-size Tensor 输入正常返回正确结果，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker: 无
- Dockerfile 或镜像地址: 暂无
- Paddle 来源: `PaddlePaddle/Paddle` source checkout at `base_commit`，需要源码编译。
- OS / Python / CUDA / cuDNN / 其他关键依赖: Linux CPU + Python + numpy；编译需要 CMake、GCC；不要求 CUDA/cuDNN（CPU 编译即可验证）。
- 硬件: CPU 即可（编译和测试均不需要 GPU）。
- patch 类型: 含 C++ kernel 修改（模板头文件，CPU/GPU 双端），需要重新编译 Paddle。
- 最小测试命令: `bash tests/test.sh`
- 是否有 oracle 日志: 无

## 7. 风险自查

- 泄露风险: 正式 `instruction.md` 只描述「multi_dot 对 0-size Tensor 输入的行为异常」，不指出具体 `numel() == 0` 分支逻辑或具体代码位置。
- 环境风险: 中。任务涉及 C++ kernel 修改，需要源码编译 Paddle，编译时间较长。
- flaky 风险: 低。测试使用固定的 0-size Tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险: 低。该 PR 目标集中在 `multi_dot` 的前向和反向 kernel 的 0-size 早期返回，测试明确指向 `TestMultiDotOp_ZeroSize1`，适合作为一个独立样本。
- 其他不确定点: 完整任务包阶段应确认新增 F2P 在 `base_commit` 编译后确实失败。注意该 PR 修改的是模板头文件 `multi_dot_kernel_impl.h`，CPU 和 GPU kernel 都会受影响。
