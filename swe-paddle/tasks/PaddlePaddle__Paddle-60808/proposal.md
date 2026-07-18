# Task Proposal: PaddlePaddle__Paddle-60808

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-60808`
* Issue 链接：https://github.com/PaddlePaddle/Paddle/issues/60780
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/60808
* PR 标题：`[BugFix]Fix broadcast_to bug while shape containing Zero-Dim`
* `base_commit`：`161551046fd4c2b8a4ce19eb50fd6f5f0eeb5645`
* merged 时间：`2024-01-16`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

修复 `paddle.broadcast_to` 在 static graph 或 dynamic-to-static 场景下无法接受包含 0-D Tensor dimension 的 `shape` list 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已关闭的 bug-report issue #60780 和已合入的 Paddle BugFix PR，不是合成任务。
* **代表性**：它覆盖 Tensor-provided shape、0-D Tensor、broadcasting semantics、static graph 和 dynamic-to-static API 行为。
* **边界清楚**：目标行为集中在 `broadcast_to` 对包含 0-D Tensor dimension 的 `shape` list 的支持，production change 集中在 `python/paddle/tensor/manipulation.py` 一个 Python 文件。
* **非平凡性**：这个任务不只是移除触发异常的 assertion。正确修复还需要区分 integer dimension、0-D Tensor dimension 和完整 1-D shape Tensor，并保证 graph construction、execution 和 broadcasting semantics 正确。
* **区分度信号**：Base 会在 graph construction 阶段稳定抛出与 issue 一致的 `AssertionError`，Solution 能完成 execution 并产生正确 output，可以区分简单绕过检查与正确处理 Tensor-provided shape 的实现。

## 4. 任务类型和标签

* 任务类型：`bug_fix`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[broadcast_to, expand, 0-D-Tensor, shape, static_graph, dy2static, python_api]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/legacy_test/test_broadcast_to_zero_dim_shape.py`
* 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，integer-list shape 和 1-D shape Tensor 相关测试应 pass，`shape` list 包含 0-D Tensor dimension 的测试应 fail。
* 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
* P2P 候选：integer-list shape 和完整 1-D shape Tensor 用例可作为回归护栏。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only production change
* 环境建议：使用已安装 Paddle runtime，并从 source checkout overlay 目标函数，避免历史完整模块与当前 wheel 不兼容
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：已完成 Base P2P/F2P 和 Solution P2P/F2P 验证

## 7. 风险自查

* 泄露风险：正式 `instruction.md` 应描述 trigger condition、observable failure 和 expected behavior，不指出内部 branch、assertion 或修改方式。
* 环境风险：验证仅 overlay 目标 Python functions，不覆盖整个历史 `manipulation.py`。
* flaky 风险：测试使用固定 shape、固定 input values 和 exact array comparison。
* 拆分风险：该 PR 的目标集中在 `broadcast_to` 的 Tensor-provided shape handling，适合作为一个样本。
