# Task Proposal: PaddlePaddle__Paddle-74221

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-74221`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74221
- PR 标题：`[0-size Tensor No.169] Add 0-size Tensor support for paddle.nn.functional.fold`
- `base_commit`：`3bfdd753fa54582b0a2ab6b47e4ea8092cec8187`
- merged 时间：`2025-07-26`
- 你的身份：原 PR 作者

## 2. 问题一句话

`paddle.nn.functional.fold` 在输入为 0-size tensor 时缺少显式校验，导致后续计算出现不可预期的错误，需要添加 0-size tensor 的断言检查并抛出明确的 `AssertionError`。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该问题来自 Paddle 的「0-size Tensor 机制建设」系列任务，是真实研发需求，目标是为 `paddle.nn.functional.fold` 补齐 0-size tensor 的边界处理。
- **代表性**：覆盖 Python API 层面的算子输入校验，涉及 0-size tensor 的早期断言和错误信息规范化，是 Paddle API 算子机制增强的典型样本。
- **边界清楚**：目标仅限 0-size tensor 输入时的断言检查，当 `math.prod(x.shape) == 0` 时应抛出 `AssertionError`；正向非零尺寸输入不应受影响。
- **非平凡性**：修复需要在 `common.py` 中为 `fold` API 添加 `math.prod(x.shape) > 0` 的断言，涉及对 tensor shape 的语义理解和错误信息规范化，不是简单机械修改。
- **回归护栏明确**：目标 F2P 可覆盖 0-size tensor 输入的 `fold` 调用；同文件中已有的 `TestFoldOpError` 等标准错误处理测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, nn_functional, fold, 0-size_tensor, input_validation]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：
  - `test/legacy_test/test_fold_op.py`（`TestFoldOpError` 中的 `test_zero_size` 子测试）
- P2P 候选：同文件中已有的 `TestFoldOpError` 等标准错误处理测试用例。
- 修复前预期：`base_commit` + `tests/test.patch` 后，0-size tensor 输入的 `fold` 调用不会抛出 `AssertionError`（缺少校验），导致 `test_zero_size` 测试失败。
- 修复后预期：继续应用 `solution/code.patch` 后，0-size tensor 输入会触发 `AssertionError`，`test_zero_size` 测试通过，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker：无
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，纯 Python 修改可直接 patch。
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：可由 verifier 选择与 base 兼容的 CPU wheel 或本地源码环境；proposal 阶段不固定 wheel URL。
- OS / Python / CUDA / cuDNN / 其他关键依赖：Linux CPU + Python + numpy + pytest；不要求 CUDA/cuDNN。
- 硬件：CPU 即可。
- patch 类型：纯 Python 修改 + Python legacy test，无需 C++ rebuild。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：无；由 SWE-Paddle verifier 记录 Run/Test/Fix 结果。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述「fold 对 0-size tensor 输入缺少边界校验」，不指出具体 `math.prod(x.shape) > 0` 断言逻辑或具体代码位置。
- 环境风险：低。任务为 Python-only，无需特殊镜像、外部服务或不可固定下载。
- flaky 风险：低。测试使用固定的 0-size tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险：低。该 PR 目标集中在 `common.py` 中 `fold` API 的 0-size 断言检查，测试也明确指向 `TestFoldOpError` 的零尺寸分支，适合作为一个独立样本。
- 其他不确定点：完整任务包阶段应确认新增 F2P（`test_zero_size`）在 `base_commit` 上确实失败，并选择同文件中已有的 `TestFoldOpError` 等标准测试用例作为在 base 与修复后都稳定通过的 P2P nodeid。
