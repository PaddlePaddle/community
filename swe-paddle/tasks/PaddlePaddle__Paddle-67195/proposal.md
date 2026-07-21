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

修复 Pipeline Parallel batched P2P communication 未在发送前检查 outgoing Tensor 中的 NaN/Inf，导致 invalid Tensor 已进入 communication path 后才抛出异常的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Distributed Strategy BugFix PR，不是合成任务。
- **代表性**：它覆盖 Pipeline Parallel、batched P2P communication、NaN/Inf checker，以及 communication side-effect ordering。
- **边界清楚**：目标行为集中在 outgoing Tensor 进入 communication path 前的有效性检查，production change 仅涉及 `p2p_communication.py` 中的 Python control flow。
- **非平凡性**：这个任务不只是增加 NaN/Inf 判断。正确修复还需要调整 batched operations 的检查顺序，确保 invalid outgoing Tensor 被发现时尚未发生任何 communication side effect。
- **确定性**：verifier 可以使用 controlled communication doubles 记录调用顺序，不依赖 GPU、NCCL 或真实 distributed process group。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[pipeline_parallel, p2p, distributed, nan_inf, communication_ordering, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_pp_nan_checker_before_send.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，finite send 和 receive-only 相关测试应 pass，single invalid send 与 mixed receive / invalid-send batch 相关测试应 fail。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：finite send 和 receive-only 用例可作为回归护栏，确保正常 send / receive behavior 不受影响。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production change
- 环境建议：使用已安装 Paddle runtime，通过 AST 加载 source checkout 中的目标 function，并使用 controlled P2P communication doubles
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述 invalid outgoing Tensor 必须在 communication side effect 前被拒绝，不直接指出内部 loop transformation 或具体修改行。
- 环境风险：测试不依赖 GPU、NCCL、rank launcher、真实 distributed process group 或 network interface。
- flaky 风险：controlled communication doubles 可以确定性地记录调用顺序，不执行真实 distributed communication。
- 拆分风险：该 PR 的目标集中在 Pipeline Parallel P2P communication 的 NaN/Inf pre-send checking，适合作为一个样本。
