# Task Proposal: PaddlePaddle__Paddle-65724

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-65724`
- Issue 链接：https://github.com/PaddlePaddle/Paddle/issues/48964
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/65724
- PR 标题：`[Bugfix] fix dataloader when setting persistent_workers=True`
- `base_commit`：`9a4caad68bca019e85847eb99da57f060e01caa5`
- merged 时间：`2024-07-17`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 `DataLoader` 在启用 `persistent_workers=True`、提前结束 epoch 并复用 iterator 时，batch structure metadata 失配导致后续 iteration 崩溃的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 issue #48964 和已合入的 Paddle BugFix PR，不是合成任务。
- **代表性**：它覆盖 DataLoader lifecycle、background prefetch、persistent workers、producer/consumer synchronization 和 nested batch restoration。
- **边界清楚**：production change 仅修改 `python/paddle/io/dataloader/dataloader_iter.py`。
- **非平凡性**：来源 issue 的用户可观察现象是后续 epoch 在 `_restore_batch` 前出现 `IndexError: pop from empty list`。
- **区分度信号**：verifier 将实际竞态约束拆解为可重复的 thread handoff 与 iterator reset invariants，避免依赖调度时序触发 flaky crash。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[dataloader, persistent_workers, multiprocessing, threading, prefetch, batch_structure]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_dataloader_persistent_workers_structure.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，ordinary FIFO structure handoff P2P 应 pass，background thread structure handoff 和 persistent-worker reset channel F2P 应 fail。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：ordinary FIFO structure handoff 用例可作为回归护栏。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production change
- 环境建议：该样本涉及 sparse CPU kernel，不能只依赖已有 wheel；需要 source build 或等价的已编译环境
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 描述 trigger condition、observable failure 和 expected behavior，不指出具体容器类型或修改行。
- 环境风险：只加载目标 iterator classes，复用当前安装环境中的 Paddle dependencies。
- flaky 风险：不依赖真实多进程调度时序；测试直接验证 producer/consumer 和 reset 的必要并发契约。
- 拆分风险：该 PR 的目标集中在 persistent workers 场景下的 batch structure metadata handoff 与 iterator reset，适合作为一个样本。
