# Task Proposal: PaddlePaddle__Paddle-56705

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-56705`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/56705
- PR 标题：`[BugFix]Fix memory leak in mplayers`
- `base_commit`：`23955fcfab3ecf5bfe4be9d3a4543cb0d9c7c377`
- merged 时间：`2023-08-29`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 dynamic graph model-parallel identity 和 all-reduce 操作重复调用时产生的 Python-side runtime type 累积与长期内存增长问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle PR #56705，不是合成任务。
- **代表性**：它覆盖 distributed model parallel、dynamic graph、autograd lifecycle 和 communication contract。
- **边界清楚**：production change 集中在 `python/paddle/distributed/fleet/layers/mpu/mp_ops.py`，目标行为可以由独立测试直接暴露。
- **非平凡性**：正确修复既要消除重复创建 runtime type 的问题，也要保持 identity 和 all-reduce 原有的 forward/backward behavior。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, fleet, model_parallel, autograd, dynamic_graph]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr56705_mp_ops_pylayer_lifecycle.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，已有 identity/all-reduce communication contract 测试应 pass，重复调用产生多个 distinct `PyLayer` runtime type 的测试应 fail。
- 修复后预期：继续应用 `solution/code.patch` 后，已有行为测试和 runtime type reuse 测试均应 pass。
- P2P 候选：identity 和 all-reduce 原有的 forward/backward communication behavior。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用可运行 `pytest` 的 Python 环境，通过 AST overlay 执行 checkout 中的目标控制流，并使用 controlled doubles 提供依赖；无需 Paddle source build 或 distributed launch
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述目标行为和验收标准，不直接给出 production code 的具体修改方式。
- 环境风险：测试不依赖历史 Paddle 模块的完整 import，通过 AST overlay 隔离版本兼容问题。
- flaky 风险：测试不测真实 RSS、不启动多进程、不依赖 GC timing，而是验证稳定的 runtime type reuse contract。
- 拆分风险：identity 和 all-reduce 属于同一 runtime lifecycle 问题，并由同一 production file 中的修改共同修复，适合作为一个样本。
