# Task Proposal: PaddlePaddle__Paddle-71525

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-71525`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/71525
- PR 标题：`[SOT] Add JSON dumping and Base64 encoding for log output`
- `base_commit`：`2ab13a83ce55baea53d856bf3a89cc6e490c35d6`
- merged 时间：`2025-03-11T11:14:05Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

SOT 收集的 breakgraph reason 和 subgraph 信息只能生成面向人工阅读的摘要，无法输出并恢复可供工具稳定消费的序列化报告。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 PaddlePaddle 已合入的 SOT 工程 PR，并使用原 PR 新增的完整测试文件作为行为 oracle。
- **代表性**：覆盖诊断信息的结构化导出、异常类型恢复、子图元数据往返以及环境开关控制。
- **边界清楚**：生产改动限定在三个 SOT Python 文件，测试集中于一个新增测试文件，Base 与 Gold 由相邻 Git 提交精确界定。
- **非平凡性**：改动包含通用序列化协议、两类报告的分类与恢复逻辑、信息对象相等性和编译阶段数据规范化，production diff 为 161 行。
- **环境友好性**：测试在 CPU 上运行，不需要 Torch、GPU、CUDA、网络或外部数据集。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[SOT, dynamic-to-static, diagnostics, serialization, logging]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/sot/test_info_collect.py`
- 修复前预期：三个原 PR 测试分别因缺少通用序列化、breakgraph reason 报告和 subgraph 报告能力而失败，完整 `tests/test.sh` 返回非零。
- 修复后预期：字典、breakgraph reason 和 subgraph 信息均可完成序列化与恢复，三个测试全部通过。
- P2P 候选：`test/sot/test_envs.py::TestPEP508LikeEnvironmentVariable::test_PEP508_like_env_get`，用于保护现有的 SOT 信息收集环境配置解析行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 Python 3.11 和兼容的 Paddle 3.0 CPU 安装，加载 checkout 中的纯 Python SOT 包，并采用 `STRICT_MODE=1`、`MIN_GRAPH_SIZE=0` 配置。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明只描述外部可观察的报告和往返行为，不包含 Gold patch 的具体代码组织或修改步骤。
- 环境风险：SOT Python 包需要与时代匹配的 Paddle 安装配合；交叉验证固定使用 Python 3.11 与兼容的 CPU Paddle 3.0 包。
- flaky 风险：原测试虽然生成随机字符串和整数，但只比较同一轮序列化前后的结构，不依赖随机值本身、网络或并发时序。
- 拆分风险：测试补丁使用原 PR 新增测试文件的完整 diff，生产补丁只包含三个 production 文件，并逐个进行 Gold blob 校验。
