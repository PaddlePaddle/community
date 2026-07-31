# Task Proposal: PaddlePaddle__Paddle-79167

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-79167`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79167
* PR 标题：`[API Compatibility] Add alias for paddle.random.initial_seed`
* `base_commit`：`c34031973911346f8cd98717583577f61adcf0b1`
* merged 时间：`2026-05-29`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

为 `paddle.random.initial_seed` 增加等价的顶层公共入口 `paddle.initial_seed`，并确保该名称能够通过 Paddle 顶层命名空间访问和发现。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：任务来自已合入的 Paddle API Compatibility PR，不是人工构造的需求。
* **类型多样性**：该任务属于公共 API alias 新增，可补充 SWE-Paddle 中以 bug fix 为主的任务类型。
* **边界清楚**：production change 仅涉及 Paddle 顶层命名空间中的 API 引入和公开名称发现，改动范围集中。
* **可观察性强**：可以直接验证 `paddle.initial_seed` 是否可访问、是否与 `paddle.random.initial_seed` 指向同一个函数对象，以及是否能通过顶层 API 枚举发现。
* **环境友好性**：任务仅涉及 Python 层的导入和模块属性行为，可以在 CPU 环境中直接验证，无需编译 Paddle 源码或加载完整的 native runtime。

## 4. 任务类型和标签

* 任务类型：`api_addition`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[public_api, api_alias, initial_seed, module_export, python_only]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr79167_initial_seed_alias.py`
* 修复前预期：现有 `seed`、`manual_seed` 和 `paddle.random.initial_seed` 的相关测试应通过；顶层 `paddle.initial_seed` 的可访问性、对象身份和公开发现测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，全部目标测试均应通过；`paddle.initial_seed` 应能够从顶层命名空间访问，并与 `paddle.random.initial_seed` 指向同一个函数对象。
* P2P 候选：`paddle.seed`、`paddle.manual_seed` 和 `paddle.random.initial_seed` 的对象关系及现有调用行为保持不变。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述顶层 API 的可访问性、对象身份和公开发现行为，不指出具体修改行或导入语句。
* 环境风险：AST overlay 避免导入历史 checkout 中完整 Paddle package 所需的 native extension。
* flaky 风险：测试仅验证确定性的模块属性、对象 identity 和 `dir()` 结果，不涉及随机数结果、并发或异步时序。
* 拆分风险：顶层 alias 和公开名称发现共同构成 `paddle.initial_seed` 的完整公共 API 暴露目标，适合作为一个任务。
