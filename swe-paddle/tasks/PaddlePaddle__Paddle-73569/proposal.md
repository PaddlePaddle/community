# SWE-Paddle Task Proposal: PaddlePaddle__Paddle-73569

## 1. 来源信息

- Instance ID: `PaddlePaddle__Paddle-73569`
- PR 链接: https://github.com/PaddlePaddle/Paddle/pull/73569
- PR 标题: `[Accuracy diff No.104] Fix accuracy diff for paddle.matmul API`
- Base commit: `434044ec20095341e74558c4612a0fe62fcc6508`
- Gold commit: `1399f8e514c134020f260ad3a79bc889dde810b4`
- Merged at: 2025-06-27
- 你的身份: contributor

## 2. 问题一句话

`paddle.matmul` 在 `y` 为 1-D 张量且 `transpose_y=True` 时，梯度 kernel 对 `transpose_y` 标志处理不正确，导致梯度计算结果与预期不符。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**: 来自 Paddle「精度差异修复」系列任务，是真实研发需求。
- **代表性**: 覆盖 C++ kernel 层面的 matmul 梯度计算边界情况处理，涉及 CPU 和 XPU 两端 kernel。
- **边界清楚**: 目标仅限 `y` 为 1-D 且 `transpose_y=True` 时的特殊处理；其他情况不应受影响。
- **非平凡性**: 修复需要在 `MatmulGradKernel` 中添加对 `!transpose_x && transpose_y && y.dims().size() < 2` 的检查，涉及对 matmul 数学语义和 1-D 张量特殊处理的理解。
- **回归护栏明确**: 目标 F2P 可覆盖 1-D `y` 张量且 `transpose_y=True` 的 matmul 测试；同文件中已有的标准测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型: `bug_fix`
- 执行后端: `cpu`
- 设备范围: `cpu_only`
- 模块标签: `[operator_kernel, matmul, accuracy_diff, cpu_kernel, xpu_kernel]`

## 5. 验证思路

- 目标测试命令: `bash tests/test.sh`
- 目标测试文件:
  - `test/legacy_test/test_matmul_v2_op.py`（`TestMatMulOp_trans_y`）
- P2P 候选: 同文件中已有的 `TestMatMulV2Op` 等标准 matmul 算子测试用例。
- 修复前预期: `base_commit` + `tests/test.patch` 后，1-D `y` 张量且 `transpose_y=True` 的测试失败（梯度检查不通过）。
- 修复后预期: 继续应用 `solution/code.patch` 并重新编译后，梯度计算正确，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker: 无
- Dockerfile 或镜像地址: 暂无
- Paddle 来源: `PaddlePaddle/Paddle` source checkout at `base_commit`，需要源码编译。
- OS / Python / CUDA / cuDNN / 其他关键依赖: Linux CPU + Python + numpy；编译需要 CMake、GCC；不要求 CUDA/cuDNN（CPU 编译即可验证）。
- 硬件: CPU 即可（编译和测试均不需要 GPU）。
- patch 类型: 含 C++ kernel 修改（CPU/XPU 双端），需要重新编译 Paddle。
- 最小测试命令: `bash tests/test.sh`
- 是否有 oracle 日志: 无

## 7. 风险自查

- 泄露风险: 正式 `instruction.md` 只描述「matmul 在 y 为 1-D 且 transpose_y=True 时的精度问题」，不指出具体 `!transpose_x && transpose_y && y.dims().size() < 2` 分支逻辑或具体代码位置。
- 环境风险: 中。任务涉及 C++ kernel 修改，需要源码编译 Paddle，编译时间较长。
- flaky 风险: 低。测试使用固定种子的随机数据，不依赖随机数差异或多设备同步。
- 拆分风险: 低。该 PR 目标集中在 matmul 梯度 kernel 的 1-D y 张量特殊处理，测试明确指向 `TestMatMulOp_trans_y`，适合作为一个独立样本。
- 其他不确定点: 完整任务包阶段应确认新增 F2P 在 `base_commit` 编译后确实失败。
