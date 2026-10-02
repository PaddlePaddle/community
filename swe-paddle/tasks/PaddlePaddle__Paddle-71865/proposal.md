# Task Proposal: PaddlePaddle__Paddle-71865

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-71865`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/71865
- PR 标题：`[SOT] Support builtin function \`super\``
- `base_commit`：`8c6c18cbe059e452bb18209f83b9dbc6c3d7d436`
- merged 时间：`2025-03-28T12:44:35Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

SOT 不能在完整图中正确执行 Python 内置 `super()` 的方法调用、属性访问和继承解析语义。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 PaddlePaddle 已合入的 SOT 工程 PR，并使用原 PR 自带测试作为行为 oracle。
- **代表性**：覆盖 dynamic-to-static 前端对 Python 对象模型、MRO、方法绑定和版本相关字节码的处理。
- **边界清楚**：生产改动限定在七个 SOT Python 文件，测试集中于一个新增文件，Base 与 Gold 由相邻 Git 提交精确界定。
- **非平凡性**：改动横跨 opcode executor、variable dispatch、基础变量、callable、container 和 iterator，生产 diff 超过 100 行。
- **环境友好性**：测试在 CPU 上运行，不需要 Torch、GPU、CUDA、网络或外部数据集。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[SOT, dynamic-to-static, Python super, bytecode, method binding]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/sot/test_23_super.py`
- 修复前预期：`super()` 的单继承调用测试失败，完整 `tests/test.sh` 返回非零。
- 修复后预期：原 PR 新增测试全部通过，完整 `tests/test.sh` 返回零。
- P2P 候选：`test/sot/test_call_object.py::TestCallObject::test_simple`，用于保护已有对象调用、动态属性查找和方法绑定行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用与该历史提交兼容的 Paddle 3.0 CPU 安装，并让 checkout 中的 `python/paddle/jit/sot` 作为运行时 SOT 包。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明只描述 Python 可观察行为，不包含 Gold patch 的具体实现步骤。
- 环境风险：SOT 对 Python 字节码版本敏感；交叉验证固定使用 Python 3.11 和兼容的 Paddle 3.0 CPU 包。
- flaky 风险：测试为确定性单进程用例，不依赖网络、并发时序或随机数据。
- 拆分风险：测试补丁使用原 PR 新增测试的完整 diff，生产补丁只包含七个生产文件，并通过 Gold blob 校验。
