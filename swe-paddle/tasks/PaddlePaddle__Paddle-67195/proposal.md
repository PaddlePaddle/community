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

修复 Pipeline Parallel batched P2P communication 在 invalid outgoing Tensor 已进入 communication path 后才报告 NaN/Inf 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle Distributed Strategy BugFix PR，不是合成任务。
- **代表性**：它覆盖 Pipeline Parallel、batched P2P communication、NaN/Inf checker 和 communication side-effect ordering。
- **边界清楚**：production change 仅修改 `p2p_communication.py` 一个 Python 文件，共 `+6/-13`。
- **可观察性**：Base 会在抛出 `ValueError` 前调用 communication operation；Solution 在任何 communication side effect 前拒绝 invalid Tensor。
- **确定性**：verifier 使用 controlled communication doubles 记录调用顺序，不依赖 GPU、NCCL 或真实 distributed process group。
- **回归护栏**：finite send 和 receive-only behavior 在 Base 与 Solution 上均保持通过。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[pipeline_parallel, p2p, distributed, nan_inf, communication_ordering, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_pp_nan_checker_before_send.py`
- 修复前预期：
  - finite send 与 receive-only P2P 应 pass
  - single invalid send F2P 应 fail
  - mixed receive/invalid-send batch F2P 应 fail
  - 完整 `tests/test.sh` 应 fail
- 修复后预期：
  - P2P 与两个 F2P 均应 pass
  - target production file 的 Git blob 应与 `gold_commit` 完全一致
  - 完整 `tests/test.sh` 应 pass

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production change
- 环境建议：使用已安装 Paddle runtime，通过 AST 加载 source checkout 中的目标 function，并使用 controlled P2P doubles
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由本地 Base/Solution cross validation 生成

## 7. 风险自查

- 泄露风险：`instruction.md` 描述 observable ordering contract，不指出内部 loop transformation 或具体修改行。
- flaky 风险：测试不启动真实 distributed communication，调用顺序完全确定。
- 环境风险：不依赖 GPU、NCCL、rank launcher 或 network interface。
- 误判风险：verifier 检查真实 communication side-effect log，而不是只检查 exception type。
- 投机风险：两个 F2P 分别覆盖 single send 与 mixed operation batch。
