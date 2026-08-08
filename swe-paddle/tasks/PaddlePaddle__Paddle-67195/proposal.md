# Task Proposal: PaddlePaddle__Paddle-67195

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-67195`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/67195
- PR 标题：`[BugFix] Fix pp nan checker before send`
- `base_commit`：`87d69ba93e5db77d9c0647d5954bd43a7fcb5ea5`
- merged 时间：`2024-08-09`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 Pipeline Parallel 批量 P2P 通信未在发送开始前检查待发送 Tensor 中的 NaN/Inf，导致异常数据进入通信流程后才抛出错误的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自 Paddle 已合入的分布式训练 BugFix PR，并非人工构造的问题。
- **代表性**：该任务涉及 Pipeline Parallel、批量 P2P 通信、NaN/Inf 检查，以及数据检查与通信操作之间的执行顺序。
- **边界清楚**：目标行为是确保待发送 Tensor 在进入通信流程前完成有效性检查，生产代码改动仅涉及 `p2p_communication.py` 中的 Python 控制流程。
- **非平凡性**：该任务并非简单增加一次 NaN/Inf 判断。正确修复还需要保证检查发生在整批 P2P 操作启动之前，避免异常数据触发任何通信操作。
- **确定性**：verifier 可以使用可控的通信替身记录调用顺序，无需依赖 GPU、NCCL 或真实的分布式进程组。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[pipeline_parallel, p2p, distributed, nan_inf, communication_ordering, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_pp_nan_checker_before_send.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，正常发送和仅接收相关测试应通过；单个异常发送以及同时包含接收和异常发送的批次测试应失败。
- 修复后预期：继续应用 `solution/code.patch` 后，所有目标测试均应通过。
- P2P 候选：正常发送和仅接收用例可作为回归测试，用于确认现有发送和接收行为未受到影响。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production change
- 环境建议：使用已安装的 Paddle runtime，通过 AST 加载 source checkout 中的目标函数，并使用可控的 P2P 通信替身完成验证。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 仅描述异常 Tensor 必须在通信开始前被拒绝，不直接说明内部循环应如何调整，也不指出具体修改位置。
- 环境风险：测试不依赖 GPU、NCCL、rank launcher、真实 distributed process group 或 network interface。
- flaky 风险：可控的通信替身能够确定性地记录调用顺序，且不会执行真实的分布式通信。
- 拆分风险：该 PR 的目标集中在 Pipeline Parallel P2P 通信的发送前 NaN/Inf 检查，问题范围完整且独立，适合作为一个样本。
