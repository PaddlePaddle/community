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

修复动态图模式下 model-parallel `identity` 和 `all-reduce` 被反复调用时产生的运行时类型累积与内存泄漏问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自 Paddle 已合入的 PR #56705，并非人工构造的问题。
- **代表性**：该任务涉及 distributed model parallel、动态图、自动求导生命周期，以及模型并行操作的前向和反向通信行为。
- **边界清楚**：production change 集中在 `python/paddle/distributed/fleet/layers/mpu/mp_ops.py`，问题范围明确，相关行为可以通过独立测试稳定验证。
- **非平凡性**：修复不仅需要消除重复调用造成的运行时类型累积，还必须保持 `identity` 和 `all-reduce` 原有的前向、反向及通信行为不变。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, fleet, model_parallel, autograd, dynamic_graph]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr56705_mp_ops_pylayer_lifecycle.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`identity` 和 `all-reduce` 原有通信行为的测试应通过；验证重复调用是否持续产生不同 `PyLayer` 运行时类型的测试应失败。
- 修复后预期：继续应用 `solution/code.patch` 后，原有行为测试和运行时类型复用测试均应通过。
- P2P 候选：`identity` 和 `all-reduce` 原有的前向、反向及通信行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用能够运行 `pytest` 的 Python 环境，通过 AST overlay 加载 source checkout 中的目标逻辑，并使用可控的测试替身补充运行依赖；无需编译 Paddle 源码，也无需启动真实的分布式任务。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 仅描述模型并行操作重复调用时的内存泄漏现象及预期行为，不直接说明 `PyLayer` 的定义位置、类型复用方式或具体代码修改方案。
- 环境风险：测试不依赖历史版本 Paddle 模块的完整导入，通过 AST overlay 隔离源码版本和运行环境之间的兼容性问题。
- flaky 风险：测试不直接测量真实 RSS，不启动多进程，也不依赖垃圾回收时机，而是验证重复调用过程中运行时类型数量是否保持稳定。
- 拆分风险：`identity` 和 `all-reduce` 的问题具有相同成因，并由同一 production file 中的相关修改共同解决，适合作为一个完整样本。
