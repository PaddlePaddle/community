# Task Proposal: PaddlePaddle__Paddle-74439

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-74439`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74439
- PR 标题：`[API compatibility] add paddle.ravel`
- `base_commit`：`cb81162732f15ae02e82b07f8462e04b093c2464`
- merged 时间：`2025-08-08`
- 你的身份：原 PR 作者 / reviewer / 熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为 Paddle 增加可从公共命名空间访问的 `ravel` API，使任意 rank Tensor 都能按元素顺序完整展平成一维结果。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已经合入 Paddle `develop` 的 API compatibility PR #74439。
- **代表性**：覆盖 Python Tensor API、公共命名空间导出以及 shape/rank 边界行为。
- **边界清楚**：production 修改集中在三个 Python 文件，原 PR 测试集中验证 `ravel` 前向、反向及不同 rank 输入。
- **非平凡性**：除了新增接口本身，还需要正确处理 scalar、1D、multi-dimensional 和 empty tensor，并保持已有 `flatten` 行为。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[tensor, manipulation, api_compatibility, flatten, python_api]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr74439_ravel.py`
- 修复前预期：已有 `flatten` dynamic-path regression 测试通过；`ravel` 行为和公共导出测试失败。
- 修复后预期：已有 `flatten` 测试继续通过，`ravel` 对 scalar/1D/multi-dimensional/empty 输入以及公共导出测试全部通过。
- P2P 候选：已有 `flatten` 在 dynamic/PIR 路径上的轴归一化和 native 调用行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用带 `pytest` 和 `numpy` 的 Python 环境，通过 AST overlay 执行 checkout 中真实函数控制流；无需 Paddle source build
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述公共 API 行为、边界输入和兼容性，不给出 Gold 的精确代码修改步骤。
- 环境风险：测试不 import 历史 Paddle wheel，而是执行 checkout 中的目标 Python 控制流。
- flaky 风险：测试不依赖随机数、GPU、设备调度或异步行为。
- 拆分风险：公共导出和完整展平属于同一个 `ravel` API contract，适合作为单一任务验证。
