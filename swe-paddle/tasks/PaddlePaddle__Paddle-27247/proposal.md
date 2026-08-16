# Task Proposal: PaddlePaddle__Paddle-27247

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-27247`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/27247
- PR 标题：`move DataLoader._worker_loop to top level`
- `base_commit`：`aae41c6fca67be6a090d4f83bdf6160737d15162`
- merged 时间：`2020-09-14T05:51:32Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

多进程 DataLoader 在 `spawn` 环境中无法序列化 worker 或 reader 的启动任务，导致子进程创建失败。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源 PR 记录了 `paddle.distributed.spawn` 与多进程 DataLoader 组合使用时的真实失败堆栈。
- **代表性**：问题属于 Python multiprocessing 在跨平台 DataLoader 中常见的 `spawn` 兼容问题。
- **边界清楚**：production change 仅涉及 DataLoader worker 与 legacy reader 两条进程启动路径。
- **非平凡性**：修复需要同时保证进程入口可序列化，并保持 worker 创建、进程注册和队列协作契约不变。
- **环境友好性**：行为测试使用 checkout 源码的 AST overlay 和受控 doubles，执行真实进程启动控制流与 multiprocessing 序列化器，无需构建旧版 Paddle。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[dataloader, multiprocessing, distributed, spawn]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr27247_dataloader_spawn_pickle.py`
- 修复前预期：两条既有启动流程 P2P 通过；worker 和 reader 的启动任务在 multiprocessing 序列化阶段分别失败，并出现 `_thread.lock` 无法 pickle 的错误。
- 修复后预期：所有 P2P/F2P 通过，完整目标测试文件通过。
- P2P 候选：验证 worker 数量、daemon/start 状态、进程 PID 注册、reader queue 注册、消费线程启动，以及 worker/reader 原有的 batch handoff 和结束标记行为。
- 原 PR 测试处理：`tests/test.patch` 原样包含两份 upstream test diff；这些改动仅适配调用位置，不能单独触发原 bug，因此另加行为测试验证 `spawn` 的必要序列化契约。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 Python 3、pytest；通过 AST overlay 执行 checkout 中相关方法，依赖由 controlled doubles 提供，不要求导入或构建历史 `paddle.fluid` 包。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述可观察报错和兼容要求，未说明 Gold patch 的具体代码调整。
- 环境风险：历史 `paddle.fluid` 源码与当前 wheel 不兼容；测试通过 AST overlay 隔离该差异，并直接使用 Python multiprocessing 的序列化器。
- flaky 风险：不启动真实训练或并发 worker，只执行确定性的进程配置控制流和 `spawn` 序列化阶段。
- 拆分风险：两处修改对应 Paddle 同一 DataLoader 问题在两套既有实现中的表现，属于同一问题，不拆分为多个任务。
