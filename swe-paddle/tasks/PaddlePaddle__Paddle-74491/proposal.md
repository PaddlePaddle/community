# Task Proposal: PaddlePaddle__Paddle-74491

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-74491`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74491
- PR 标题：`[API compatibility] add new API `paddle.Tensor.requires_grad``
- `base_commit`：`01666a6667e744874d7f7c379b2649d8bae67f09`
- merged 时间：`2025-08-13`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

Paddle Tensor 在 dynamic、static 和 PIR 模式下缺少统一的 `requires_grad` 属性，导致用户无法用兼容接口读取和控制是否参与梯度计算。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源于已合入 Paddle `develop` 的 API compatibility PR。
- **代表性**：覆盖常见的 Tensor 梯度控制兼容接口，并要求多个执行模式行为一致。
- **边界清楚**：目标契约集中在 getter、setter、非 bool 拒绝以及与 `stop_gradient` 的一致性。
- **非平凡性**：需要同时覆盖 dynamic graph、static graph 和 PIR 三条 Tensor patch 路径，而不是只增加单一入口。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[tensor, autograd, api_compatibility, dynamic_graph, static_graph, pir]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr74491_requires_grad.py`
- 修复前预期：已有 `dim` 元数据行为通过；`requires_grad` property 在三个执行模式的源码路径中不可用，因此对应 F2P 失败。
- 修复后预期：三个模式均满足 `requires_grad == not stop_gradient`，布尔 setter 正确更新状态，非 bool 赋值抛出 `TypeError`，全部测试通过。
- P2P 候选：执行三个 checkout 文件中已有 `dim` 函数体，确认 shape rank 行为未退化。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 Python + pytest；通过 AST overlay 提取 checkout 中真实函数/property body 并在 controlled namespace 中执行。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：测试只验证公开梯度状态契约，不要求具体注册列表、局部变量或实现步骤。
- 环境风险：低；不导入历史 Paddle native runtime，不需要编译或 GPU。
- flaky 风险：低；全部输入为确定性 Python 对象，不依赖随机、并发或外部资源。
- 拆分风险：低；三个 production 文件共同实现同一个跨执行模式 API，适合作为一个完整任务。
