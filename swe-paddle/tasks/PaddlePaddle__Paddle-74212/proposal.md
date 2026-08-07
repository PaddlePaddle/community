# SWE-Paddle Task Proposal: PaddlePaddle__Paddle-74212

## 1. 来源信息

- Instance ID: `PaddlePaddle__Paddle-74212`
- PR 链接: https://github.com/PaddlePaddle/Paddle/pull/74212
- PR 标题: `[0-size Tensor Job2 No.51] Add 0-size Tensor support for paddle.multiplex`
- Base commit: `0f3860d981460b0b788aa50836a215f59c90e32a`
- Gold commit: `3e59330aa066d997e24ff6c5c74c19b250fae43d`
- Merged at: 2025-07-28
- 你的身份: contributor

## 2. 问题一句话

`paddle.multiplex` 在输入为 0-size Tensor 时，kernel 会因 PADDLE_ENFORCE_GT 检查每个输入 numel > 0 而报错，需要在 kernel 入口添加 0-size 早期返回逻辑。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**: 来自 Paddle「0-size Tensor 机制建设」系列任务，是真实研发需求。
- **代表性**: 覆盖 C++ kernel 层面的 0-size Tensor 边界处理，涉及 CPU/GPU 双端 kernel，需要在 kernel 入口添加 `out->numel() == 0` 的早期返回。
- **边界清楚**: 目标仅限所有输入均为 0-size 时的 kernel 早期返回；正向非零尺寸输入不应受影响。
- **非平凡性**: 修复需要在 CPU/GPU 两个 backend 的 kernel 中分别添加 `if (out->numel() == 0) return;`，涉及对 kernel 执行流程的理解和 0-size Tensor 语义的把握。
- **回归护栏明确**: 目标 F2P 可覆盖 0-size Tensor 输入的 `multiplex` 算子测试；同文件中已有的 `TestMultiplexOp` 等标准测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型: `bug_fix`
- 执行后端: `cpu`
- 设备范围: `cpu_only`
- 模块标签: `[operator_kernel, multiplex, 0-size_tensor, cpu_kernel, gpu_kernel]`

## 5. 验证思路

- 目标测试命令: `bash tests/test.sh`
- 目标测试文件:
  - `test/legacy_test/test_multiplex_op.py`（`TestMultiplexOp_ZeroSize`）
- P2P 候选: 同文件中已有的 `TestMultiplexOp`、`TestMultiplexODygrap` 等标准 multiplex 算子测试用例。
- 修复前预期: `base_commit` + `tests/test.patch` 后，0-size Tensor 输入的 `multiplex` 算子测试失败（kernel 内部 PADDLE_ENFORCE_GT 报错）。
- 修复后预期: 继续应用 `solution/code.patch` 并重新编译后，0-size Tensor 输入正常返回空 Tensor，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker: 无
- Dockerfile 或镜像地址: 暂无
- Paddle 来源: `PaddlePaddle/Paddle` source checkout at `base_commit`，需要源码编译。
- OS / Python / CUDA / cuDNN / 其他关键依赖: Linux CPU + Python + numpy；编译需要 CMake、GCC；不要求 CUDA/cuDNN（CPU 编译即可验证）。
- 硬件: CPU 即可（编译和测试均不需要 GPU）。
- patch 类型: 含 C++ kernel 修改（CPU/GPU 双端），需要重新编译 Paddle。
- 最小测试命令: `bash tests/test.sh`
- 是否有 oracle 日志: 无

## 7. 风险自查

- 泄露风险: 正式 `instruction.md` 只描述「multiplex 对 0-size Tensor 输入的行为异常」，不指出具体 `out->numel() == 0` 分支逻辑或具体代码位置。
- 环境风险: 中。任务涉及 C++ kernel 修改，需要源码编译 Paddle，编译时间较长。
- flaky 风险: 低。测试使用固定的 0-size Tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险: 低。该 PR 目标集中在 `multiplex_kernel` 的 CPU/GPU 双端 0-size 早期返回，测试明确指向 `TestMultiplexOp_ZeroSize`，适合作为一个独立样本。
- 其他不确定点: 完整任务包阶段应确认新增 F2P（`TestMultiplexOp_ZeroSize`）在 `base_commit` 编译后确实失败。
