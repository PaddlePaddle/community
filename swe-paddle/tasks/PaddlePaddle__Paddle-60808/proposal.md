# Task Proposal: PaddlePaddle__Paddle-60808

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-60808`
- Issue 链接：https://github.com/PaddlePaddle/Paddle/issues/60780
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/60808
- PR 标题：`[BugFix]Fix broadcast_to bug while shape containing Zero-Dim`
- `base_commit`：`161551046fd4c2b8a4ce19eb50fd6f5f0eeb5645`
- `gold_commit`：`e2a324cb86120d2e82d9d9dbd9400d60e3a4bc8e`
- merged 时间：`2024-01-16`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 `paddle.broadcast_to` 在 static graph 或 dynamic-to-static 场景下无法接受包含 0-D Tensor dimension 的 `shape` list 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已关闭的 bug-report issue #60780 和已合入的 Paddle BugFix PR，不是合成任务。
- **代表性**：它覆盖 Tensor-provided shape、0-D Tensor、broadcasting semantics、static graph 和 dynamic-to-static API 行为。
- **边界清楚**：production change 集中在 `python/paddle/tensor/manipulation.py` 一个 Python 文件。
- **可观察性**：Base 在 graph construction 阶段稳定抛出与 issue 一致的 AssertionError；Solution 能完成 execution 并产生正确 output。
- **回归护栏**：integer-list shape 和完整 1-D shape Tensor 在 Base 与 Solution 上均保持通过。
- **环境友好**：测试可在 CPU 环境运行，不需要 source build、GPU、distributed runtime 或外部数据集。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[broadcast_to, expand, 0-D-Tensor, shape, static_graph, dy2static, python_api]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_broadcast_to_zero_dim_shape.py`
- 修复前预期：
  - integer-list shape P2P 应 pass
  - 1-D shape Tensor P2P 应 pass
  - `shape` list 包含 0-D Tensor dimension 的 F2P 应 fail
  - 完整 `tests/test.sh` 应 fail，并且只包含上述目标失败
- 修复后预期：
  - 应用 `solution/code.patch` 后，P2P 与 F2P 均应 pass
  - 目标 production file 的 Git blob 应与 `gold_commit` 完全一致
  - 完整 `tests/test.sh` 应 pass

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production change
- 环境建议：使用已安装 Paddle runtime，并从 source checkout overlay 目标函数，避免历史完整模块与当前 wheel 不兼容
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：已完成 Base P2P/F2P、Solution P2P/F2P 和 Gold blob 验证

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述 trigger condition、observable failure 和 expected behavior，不指出内部 branch、assertion 或修改方式。
- 环境风险：验证仅 overlay 目标 Python functions，不覆盖整个历史 `manipulation.py`。
- flaky 风险：测试使用固定 shape、固定 input values 和 exact array comparison。
- 误判风险：F2P 同时验证 graph construction、execution、output shape 和 broadcasted values，不仅检查是否抛异常。
- 拆分风险：任务目标集中在 `broadcast_to` 的 Tensor-provided shape handling，适合作为一个样本。
