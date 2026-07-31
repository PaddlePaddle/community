# Task Proposal: PaddlePaddle__Paddle-74594

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-74594`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74594
* PR 标题：`[API compatibility] add broadcast_shapes api`
* `base_commit`：`20e50fb447d8e34bafb337c180ca28d77dfb82ca`
* merged 时间：`2025-08-15`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

新增 `paddle.broadcast_shapes` 公共 API，使调用方能够一次计算任意数量 shape 的共同广播结果，并在输入无法广播时得到明确的异常。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已合入的 Paddle PR #74594，不是人工构造的需求。
* **代表性**：该任务涉及 API compatibility、公共 API 导出以及多个 shape 的广播规则。
* **边界清楚**：production change 集中在 `python/paddle/__init__.py`、`python/paddle/tensor/__init__.py` 和 `python/paddle/tensor/math.py`，测试代码可以与 Gold production patch 清晰分离。
* **非平凡性**：任务需要正确处理多个 shape、无输入、单输入和空 shape，并在输入不兼容时保持明确的异常行为，同时保证现有二元 `broadcast_shape` 不受影响。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[api_compatibility, tensor, math, broadcasting, shape]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr74594_broadcast_shapes.py`
* 修复前预期：已有 `broadcast_shape` delegation P2P 应 pass；`broadcast_shapes` 的多输入、零输入和单输入测试应因 API 尚不存在而 fail。
* 修复后预期：应用 production Gold patch 后，P2P 与全部 F2P 均应 pass；多个可广播 shape 应得到正确结果，不兼容输入应继续抛出 `ValueError`。
* P2P 候选：已有 `broadcast_shape` 继续按原有方式接收两个 shape，并返回底层广播结果。
* F2P 候选：三个及以上 shape 的共同广播；零输入、单输入和 empty shape；不兼容输入的异常行为。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only
* 环境建议：有Paddle 环境，使用能够运行 `pytest` 的 Python 环境
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述公共 API 的输入、输出、异常和兼容性要求，不透露 Gold patch 的循环结构、内部函数调用方式或具体实现步骤。
* 环境风险：测试不依赖历史 Paddle wheel、GPU、CUDA、native extension 或网络环境。
* flaky 风险：测试使用固定 shape 和确定性的异常断言，不涉及随机数、并发或异步执行。
* 拆分风险：公共命名空间导出和多 shape 广播行为共同构成 `broadcast_shapes` 的完整能力，适合作为单个任务。
