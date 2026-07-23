# Task Proposal: PaddlePaddle__Paddle-79167

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79167`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79167
- PR 标题：`[API Compatibility] Add alias for paddle.random.initial_seed`
- `base_commit`：`c34031973911346f8cd98717583577f61adcf0b1`
- merged 时间：`2026-05-29`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为 `paddle.random.initial_seed` 增加等价的顶层公共 API `paddle.initial_seed`，并确保该名称参与标准公共导出。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle API Compatibility PR，不是合成需求。
- **类型多样性**：这是公共 API addition/alias 任务，而不是 bug fix。
- **边界清楚**：production 行为仅涉及 Paddle 顶层命名空间和公共导出集合。
- **可观察性强**：可以直接验证新名称是否与原 API 指向同一对象，以及是否出现在 `__all__` 中。
- **环境友好性**：Python-only、CPU-only；通过执行 checkout 中真实 import/export AST 节点即可稳定验证，无需 native source build。

## 4. 任务类型和标签

- 任务类型：`api_addition`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[public_api, api_alias, initial_seed, module_export, python_only]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr79167_initial_seed_alias.py`
- 修复前预期：现有 `seed` / `manual_seed` 导出测试通过；顶层 `initial_seed` alias 和公共导出测试失败。
- 修复后预期：继续应用 `solution/code.patch` 后，全部目标测试通过。
- P2P 候选：`seed` 仍指向原 framework random API，`manual_seed` 仍为 `seed` 的别名，且两者继续位于公共导出集合。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：从 checkout 的 `python/paddle/__init__.py` 提取并执行相关 import、alias assignment 和 `__all__` AST 节点，以 controlled module doubles 验证真实 Python 导出语义。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述公共 API contract，不指出具体修改行。
- 环境风险：AST overlay 避免导入历史 checkout 中完整 Paddle package 所需的 native extension。
- flaky 风险：测试仅验证确定性的对象 identity 和公共导出元数据。
- 拆分风险：顶层 alias 与 `__all__` 注册共同构成一个完整的公共 API 暴露目标。
