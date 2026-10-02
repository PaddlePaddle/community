# Task Proposal: PaddlePaddle__Paddle-77020

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-77020`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/77020
- PR 标题：`[SOT] Add fully \`SizeVariable\` support`
- `base_commit`：`c8f9b3d44f61dd80d2c06b40000d1c21f7dca427`
- merged 时间：`2025-12-23T17:05:27Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

SOT 不能完整保持 `paddle.Size` 在构造、索引、运算、方法调用及 symbolic shape 场景中的类型和行为语义。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 PaddlePaddle 已合入的 SOT 工程 PR，并使用原 PR 自带测试作为行为 oracle。
- **代表性**：覆盖 dynamic-to-static 执行中专用容器类型的分发、传播和运算语义，是编译前端常见的跨模块工程问题。
- **边界清楚**：生产改动限定在四个 Python 文件，测试集中于一个新增文件，Base 与 Gold 由相邻 Git 提交精确界定。
- **非平凡性**：生产代码 diff 超过 100 行且跨变量分发、可调用对象和容器抽象，不能靠单点条件判断完成。
- **环境友好性**：测试在 CPU 上运行，不需要 Torch、GPU、CUDA、网络或外部数据集。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[SOT, dynamic-to-static, paddle.Size, Python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/sot/test_sot_paddle_size.py`
- 修复前预期：原 PR 新增的 `paddle.Size` 行为测试失败，完整 `tests/test.sh` 返回非零。
- 修复后预期：新增文件中的 8 个测试全部通过，完整 `tests/test.sh` 返回零。
- P2P 候选：`test/sot/test_04_list.py::TestListMethods::test_list_add`，用于保护已有 list 加法翻译行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用与该历史提交兼容的 CPU Paddle 构建，并让 checkout 中的 `python/paddle/jit/sot` 覆盖运行时对应的纯 Python 模块。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明仅描述可观察行为；实现补丁仅保存在标准 Gold patch 中。
- 环境风险：SOT Python 源码需要与可加载的 Paddle CPU 二进制配合，交叉验证脚本负责同步 checkout 源码。
- flaky 风险：测试为本地确定性单进程用例，不依赖网络、并发时序或随机数据。
- 拆分风险：测试补丁使用原 PR 新增测试的完整 diff，生产补丁只包含四个生产文件，并通过 Gold blob 校验。
