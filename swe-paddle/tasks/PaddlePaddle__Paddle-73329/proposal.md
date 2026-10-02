# Task Proposal: PaddlePaddle__Paddle-73329

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-73329`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/73329
- PR 标题：`[SOT] Support \`dataclass\` construction and basic access`
- `base_commit`：`eb9af7cf69977981198f41ae79eebabd0e3e036f`
- merged 时间：`2025-06-15T09:50:03Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

SOT strict mode 不能完整执行 Python `dataclass` 的构造及基本实例访问语义。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 PaddlePaddle 已合入的 SOT 工程 PR，并使用原 PR 测试 diff 作为行为 oracle。
- **代表性**：覆盖 dynamic-to-static 前端对 Python 结构化对象、默认字段、side effect、属性追踪和重建的处理。
- **边界清楚**：生产改动限定在七个 SOT Python 文件，核心测试集中于 `test/sot/test_dataclass.py`，Base 与 Gold 由相邻 Git 提交精确界定。
- **非平凡性**：改动横跨 function graph、side effects、variable dispatch 和多个 variable abstraction，生产 diff 超过 300 行。
- **环境友好性**：测试在 CPU 上运行，不需要 Torch、GPU、CUDA、网络或外部数据集。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[SOT, dynamic-to-static, dataclass, side effects, Python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/sot/test_dataclass.py`
- 修复前预期：在 `STRICT_MODE=1`、`MIN_GRAPH_SIZE=0` 的标准 SOT CI 配置下，dataclass 构造测试失败，完整 `tests/test.sh` 返回非零。
- 修复后预期：原 PR 的 dataclass 测试全部通过，完整 `tests/test.sh` 返回零。
- P2P 候选：`test/sot/test_04_list.py::TestListMethods::test_list_add`，用于保护已有容器加法和变量分发行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 Python 3.11 和兼容的 Paddle CPU 安装，加载 checkout 中的纯 Python SOT 包，并采用官方 SOT CI 的 `STRICT_MODE=1`、`MIN_GRAPH_SIZE=0` 配置。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明只描述可观察行为，不包含 Gold patch 的具体实现步骤。
- 环境风险：F2P 依赖官方 SOT strict-mode 配置；交叉验证脚本显式固定该配置并适配本地 CPU 二进制接口版本。
- flaky 风险：测试为确定性单进程用例，不依赖网络、并发时序或随机命中。
- 拆分风险：测试补丁使用原 PR 三个测试文件的完整 diff，生产补丁只包含七个生产文件，并逐个进行 Gold blob 校验。
