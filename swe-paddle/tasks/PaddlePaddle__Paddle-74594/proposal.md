# Task Proposal: PaddlePaddle__Paddle-74594

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-74594`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74594
- PR 标题：`[API compatibility] add broadcast_shapes api`
- `base_commit`：`20e50fb447d8e34bafb337c180ca28d77dfb82ca`
- merged 时间：`2025-08-15`
- 你的身份：原 PR 作者 / reviewer / 熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

补齐 `paddle.broadcast_shapes` 公共 API，使调用方能够一次计算零个、一个或多个 shape 的共同广播结果，并在不兼容输入时得到明确失败。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle PR #74594，不是人工构造的需求。
- **代表性**：它覆盖 API compatibility、新公共 API 暴露以及 shape broadcasting contract。
- **边界清楚**：production change 仅涉及 `python/paddle/__init__.py`、`python/paddle/tensor/__init__.py` 和 `python/paddle/tensor/math.py`，原 PR test 可与 Gold production patch 严格分离。
- **非平凡性**：需要同时处理多 shape 逐步广播、零/单输入 identity，以及不兼容输入错误传播，同时保持已有二元 `broadcast_shape` 行为不退化。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[api_compatibility, tensor, math, broadcasting, shape]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr74594_broadcast_shapes.py`
- 修复前预期：已有 `broadcast_shape` delegation P2P 应 pass；`broadcast_shapes` 的多输入和 zero/single-input 测试因 API 不存在而 fail。
- 修复后预期：应用 production Gold patch 后，P2P 与全部 F2P 均应 pass；不兼容 shape 仍通过底层 broadcasting contract 抛出 `ValueError`。
- P2P 候选：已有 `broadcast_shape` 继续原样转发输入并返回底层结果。
- F2P 候选：三个及以上 shape 合并；零输入、单输入和 empty-shape identity；不兼容输入错误传播。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用可运行 `pytest` 的 Python 环境，通过 AST 提取 checkout 中真实 `broadcast_shape` / `broadcast_shapes` 函数并提供受控 `core.broadcast_shape` double；无需 Paddle source build
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述公共 API 的输入输出、异常和兼容性，不提供 Gold 的循环结构、内部 helper 选择或实现步骤。
- 环境风险：测试不依赖历史 Paddle wheel、GPU、CUDA、native extension 或网络。
- flaky 风险：测试为纯 Python、固定 shape 输入和确定性异常断言，无随机性和并发。
- 拆分风险：公共导出与多 shape broadcasting 是同一 API compatibility feature 的必要组成部分，适合作为单个任务。
