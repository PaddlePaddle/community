# Task Proposal: PaddlePaddle__Paddle-71747

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-71747`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/71747
- PR 标题：`[SOT] Support reduce-like operation for all iterables`
- `base_commit`：`243bad9d6a79d843b4db400c9f935a6d19ed48d0`
- merged 时间：`2025-03-18T16:55:04Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

SOT 不能对 generator 等通用 iterable 完整执行 `sum`、`max`、`min` 和 `map` 等 reduce-like 操作。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 PaddlePaddle 已合入的 SOT 工程 PR，并使用原 PR 的 generator 测试 diff 作为行为 oracle。
- **代表性**：覆盖 dynamic-to-static 前端对 iterator protocol、builtin dispatch 和 iterable 消费语义的处理。
- **边界清楚**：生产改动限定在六个 SOT Python 文件，测试集中于一个 generator 测试文件，Base 与 Gold 由相邻 Git 提交精确界定。
- **非平凡性**：改动横跨 variable dispatch、基础变量、容器变量、iterator variable 和公共工具，生产 diff 为 165 行。
- **环境友好性**：测试在 CPU 上运行，不需要 Torch、GPU、CUDA、网络或外部数据集。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[SOT, dynamic-to-static, generator, iterable, builtin dispatch]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/sot/test_22_generator.py`
- 修复前预期：原有 generator 行为和 `reduce` 测试通过；`sum`、`max`、`min`、`map` 的 generator 调用失败，完整 `tests/test.sh` 返回非零。
- 修复后预期：原 PR generator 测试全部通过，完整 `tests/test.sh` 返回零。
- P2P 候选：`TestGeneratorCommon::test_generator_simple` 和 `TestGeneratorDispatch::test_generator_reduce`。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 Python 3.11 和兼容的 Paddle 3.0 CPU 安装，加载 checkout 中的纯 Python SOT 包，并采用官方 SOT CI 的 `STRICT_MODE=1`、`MIN_GRAPH_SIZE=0` 配置。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明只描述可观察行为，不包含 Gold patch 的具体实现步骤。
- 环境风险：SOT 对 Python 字节码和运行时版本敏感；交叉验证固定使用 Python 3.11 与兼容的 CPU Paddle 包。
- flaky 风险：测试为确定性纯 Python generator 用例，不依赖网络、随机数据或并发时序。
- 拆分风险：测试补丁使用原 PR 测试文件的完整 diff，生产补丁只包含六个生产文件，并逐个进行 Gold blob 校验。
