# Task Proposal: PaddlePaddle__Paddle-74586

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-74586`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74586
- PR 标题：`[API compatibility] add scatter_add api`
- `base_commit`：`e5c11eb4ab20851a6ab76bd0a85c8650b20b0692`
- merged 时间：`2025-08-21`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

Paddle 缺少可从公共命名空间调用、并按索引在指定维度执行累加语义的 `scatter_add` API。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源于已合入 Paddle `develop` 的 API compatibility PR。
- **代表性**：覆盖常见的张量 scatter-add 兼容接口与公共 API 暴露问题。
- **边界清楚**：Base 不提供该 API，Gold 提供后可直接观察重复索引的累加结果。
- **非平凡性**：不仅需要新增接口，还必须保持指定维度、索引映射和 existing-input accumulation 的语义。
- **环境友好性**：核心 Python 控制流可通过 AST overlay 与 controlled double 稳定验证，无需完整 source build。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[api_compatibility, tensor, manipulation, scatter]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr74586_scatter_add.py`
- 修复前预期：已有 manipulation P2P 通过；`scatter_add` 行为与公共导出 F2P 失败。
- 修复后预期：P2P 与所有 F2P 均通过。
- P2P 候选：验证现有 `index_add` dynamic path 的参数传递和返回值保持不变。
- F2P 设计：通过 AST overlay 执行 checkout 中真实 `scatter_add` 函数体，并使用 controlled `put_along_axis` double 验证重复索引累加结果；另验证两个公共命名空间导出。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 pytest、NumPy 和 AST overlay；无需导入历史源码对应的完整 Paddle wheel。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：测试验证公开行为和真实函数控制流，不检查 Gold 源码字符串或内部局部变量名。
- 环境风险：低；Python-only，CPU 即可，无需 CUDA 或 source build。
- flaky 风险：低；测试使用固定 NumPy 输入和 controlled doubles，不依赖随机时序。
- 拆分风险：低；production 改动集中在 manipulation API 及两处公共导出。
