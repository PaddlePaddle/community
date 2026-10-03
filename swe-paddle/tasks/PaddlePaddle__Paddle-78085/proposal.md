# Task Proposal: PaddlePaddle__Paddle-78085

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78085`
- Issue 链接：https://github.com/PaddlePaddle/Paddle/issues/77735
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78085
- PR 标题：`fix: Implement RestrictedUnpickler to prevent RCE via pickle deserialization (CWE-502)`
- `base_commit`：`bb769229044ec937446dbdb55e04ac22d83b313b`
- merged 时间：`2026-03-13T14:57:12+08:00`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

Paddle 的模型加载路径会直接反序列化 Pickle 对象，恶意模型文件可能借此执行任意 Python 调用。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自 `paddle.load` 的真实安全缺陷和已合入修复，来源 issue 给出了明确触发条件。
- **代表性**：覆盖模型参数、state dict、静态图信息、JIT 模型和 checkpoint 元数据等常见加载入口。
- **边界清楚**：生产改动和新增测试均来自相邻的 Base/Gold 提交，测试集中在一个新增文件。
- **非平凡性**：production diff 为 268 行、横跨 11 个 Python 文件，需要同时处理安全限制和已有格式兼容性。
- **环境友好性**：核心行为测试使用内存或临时文件，可在 CPU 上稳定运行。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[framework, io, serialization, model-loading, security]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_restricted_unpickler.py`
- 修复前预期：测试收集因缺少受限制的安全反序列化能力而失败。
- 修复后预期：正常 NumPy/state-dict 数据可加载，危险调用被拒绝，DenseTensor 与大文件辅助路径保持兼容，原 PR 的 20 个测试全部通过。
- P2P 候选：`test/legacy_test/test_paddle_save_load.py::TestSaveLoadToMemory::test_dygraph_save_to_memory`，用于保护现有内存 save/load 行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：Python 3.11、兼容的 Paddle CPU 运行时，并从 checkout 加载本任务涉及的 Python 模块。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明只描述可观察的安全与兼容行为，不提供 Gold patch 的具体实现步骤。
- 环境风险：checkout 中的 Python 文件依赖已安装 Paddle 的编译运行时；交叉验证脚本只加载目标 Python 模块和辅助函数。
- flaky 风险：测试仅使用内存缓冲区与本地临时文件，不依赖网络、并发时序或随机外部状态。
- 拆分风险：测试补丁是原 PR 新增测试文件的精确 diff；生产补丁包含 PR 的全部 11 个 Python production 文件并逐一校验 Gold blob。
