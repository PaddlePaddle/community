# Task Proposal: PaddlePaddle__Paddle-36684

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-36684`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/36684
- PR 标题：`fleet support elastic scale up/down`
- `base_commit`：`9a9345fa4dc77be655811d8e484b99cb9ff5f356`
- merged 时间：`2021-11-11T06:27:42Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

Fleet 弹性训练需要接受节点数量范围，并在节点加入或退出后更新训练所需的主机、rank 和 endpoint 信息。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源于 Fleet 弹性训练对动态资源调度的实际支持需求。
- **代表性**：覆盖节点范围解析、启动条件判断和扩缩容后的环境更新。
- **边界清楚**：production change 集中在 Python 分布式 elastic launch 流程。
- **非平凡性**：需要同时处理固定节点模式、范围模式、超时等待以及扩缩容后的 rank/endpoint 连续性。
- **环境友好性**：测试使用来源 PR 的 unittest 场景和 controlled doubles，不需要启动真实 etcd、训练进程或 GPU。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, fleet, elastic, launch]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr36684_elastic_scale.py`
- 修复前预期：固定节点的 elastic enablement 继续通过；来源 PR 中范围匹配和扩缩容用例因 Base 不支持范围节点数而失败。
- 修复后预期：固定节点 P2P 与两个范围扩缩容 F2P 全部通过。
- P2P 候选：来源 PR `TestElasticInit.test_enable_elastic` 对既有 elastic enablement 的验证。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 AST overlay 执行 checkout 中的真实控制流，并直接调用来源 PR 的 unittest 方法；etcd 和设备探测使用 controlled doubles。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述外部行为，没有给出 Gold patch 的具体修改方式。
- 环境风险：无需真实 etcd、分布式集群、网络服务或 GPU。
- flaky 风险：不依赖真实节点变化、计时竞态或后台训练进程。
- 拆分风险：测试只覆盖节点范围匹配及扩缩容环境更新这一组相互依赖的行为。
