# Task Proposal: PaddlePaddle__Paddle-79633

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79633`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79633
- PR 标题：`[Distributed Strategy] Fix KV server hangs under concurrent requests`
- `base_commit`：`58354a509a8d60b2cb3cdf6ead63a6c845eefd23`
- merged 时间：`2026-08-10T12:30:53Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

分布式启动使用的 KV server 在并发注册或遇到未完成请求时可能阻塞，导致其他节点无法继续完成启动同步。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自大规模分布式启动过程中节点注册和信息同步的真实阻塞场景。
- **代表性**：覆盖 Python 网络服务的并发处理、异常连接回收和服务生命周期管理。
- **边界清楚**：production change 仅涉及 KV server，测试也只访问本机临时端口。
- **非平凡性**：修复需要同时保证并发请求、半开连接和正常停止，不是简单修改返回值或错误信息。
- **环境友好性**：原 PR 测试使用 CPU、loopback 网络和系统分配的临时端口，不需要 GPU、外部服务或数据集。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed_launch, kv_server, concurrency, networking]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_kv_server.py`
- 修复前预期：服务停止测试通过；并发请求测试和未完成请求超时测试失败。
- 修复后预期：三个原 PR 测试全部通过，服务可以并发响应、释放异常连接并正常停止。
- P2P 候选：`TestKVServerStop::test_stop_is_clean_and_idempotent_state`
- F2P 候选：`TestKVServerConcurrent::test_concurrent_put_get_prefix`、`TestKVServerRequestTimeout::test_half_open_connection_is_released_after_timeout`

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用与源码兼容的 Paddle Python 环境；测试仅使用本机 loopback 网络和临时端口。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述可观察的阻塞、恢复和停止行为，没有给出 Gold patch 的具体实现。
- 环境风险：测试依赖 `httpx`，该依赖随 Paddle launch 环境提供；不需要外部网络。
- flaky 风险：测试使用系统分配的临时端口，并设置有界等待；不依赖真实集群竞态。
- 拆分风险：PR 只解决 KV server 在并发和异常连接下阻塞这一项问题，三个测试共同验证同一服务可用性契约。
