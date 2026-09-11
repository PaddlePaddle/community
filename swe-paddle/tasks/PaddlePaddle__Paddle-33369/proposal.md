# Task Proposal: PaddlePaddle__Paddle-33369

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-33369`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/33369
- PR 标题：`ELASTIC 1 : fault tolerance`
- `base_commit`：`4b9430a1f9ac2650a6a58e061f005acf8fc12fb3`
- merged 时间：`2021-06-21T06:06:29Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

fleet collective launcher 需要在 worker 失败或节点成员变化时安全停止当前进程组，并向外返回可用于重启或重新组网的状态。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：长时间分布式训练中的 worker 失败和节点变化是实际会发生的故障场景。
- **代表性**：任务覆盖 launcher 路由、进程状态判断、节点同步和退出状态传播，是分布式控制面的典型问题。
- **边界清楚**：production 改动集中在 fleet launch 与新增 elastic manager，不涉及模型、算子或训练数值。
- **非平凡性**：需要协调多个生命周期状态，并保持未启用 elastic 和 parameter-server 模式的兼容行为；production diff 为 495 行。
- **环境友好性**：来源 PR 未提供可直接复用的 Python unit test，candidate 使用 controlled launchers 执行 checkout 中真实 manager/launch 控制流，不连接 etcd、不启动进程。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, fleet, launch, elastic]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr33369_elastic_fault_tolerance.py`
- 修复前预期：parameter-server mode 的 P2P 通过；elastic manager、失败重启、成员变化暂停和 collective 生命周期接管相关 F2P 失败。
- 修复后预期：失败 worker 返回 restart，成员变化返回 hold 并停止 launcher，collective launch 由 elastic 生命周期管理，全部测试通过。
- P2P 候选：显式选择 parameter-server mode 时仍返回原有模式。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用带 pytest 和 `six` 的 Python 环境；测试不要求 etcd、网络、GPU 或真实子进程。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 说明故障恢复的外部行为和兼容要求，不描述类名、字段名或 Gold patch 的文件拆分方式。
- 环境风险：真实 elastic 依赖 etcd；测试用 controlled doubles 隔离外部服务，同时执行 checkout 中的真实状态分支。
- flaky 风险：不使用真实多进程、网络或超时轮询，所有 worker 和 membership 状态均为确定输入。
- 拆分风险：PR 的两个 production 文件共同实现同一条 collective fault-tolerance 生命周期，无法再拆成独立问题。
