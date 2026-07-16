# Task Proposal: PaddlePaddle__Paddle-78440

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78440`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78440
- PR 标题：`[Fix] Fix paddle.cdist 0-size tensor handling: correct batch shape and stop_gradient propagation`
- `base_commit`：`399e85b7d6c76c49af8301f718690c8d24548554`
- merged 时间：`2026-03-24-15:39`
- 任务类型：`bug_fix`

- 你的身份：contributor

- 后续联系人：TBD

## 2. 问题一句话

`paddle.cdist` 在零尺寸输入下可能返回错误的高维 batch shape，并丢失输入的梯度需求。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自 Paddle 主仓已合入的真实 bug-fix。
- **边界清楚**：实现修改集中在 `paddle.cdist` 的零尺寸特殊分支。
- **外部行为明确**：修复前 shape 和 `stop_gradient` 断言稳定失败，修复后通过。
- **非平凡性**：需要同时理解 batch 广播语义和 Eager autograd 状态传播。
- **成本低**：Python-only patch，不需要重新编译 C++ core。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, cdist, linalg, zero_size, autograd, broadcasting]`

## 5. 验证思路

- 目标命令：`bash tests/test.sh`
- 目标文件：`test/legacy_test/test_cdist.py`
- P2P：`TestCdistZeroSizeGrad::test_stop_gradient_true`
- F2P：`TestCdistZeroSizeBatch4D::test_dygraph_api`
- F2P：`TestCdistZeroSizeGrad::test_stop_gradient_false`
- 修复前：P2P 通过，两个 F2P 失败。
- 修复后：三个用例均通过。

## 6. 环境与资源

- 资源需求：CPU
- GPU、网络服务和外部模型：不需要
- patch 类型：Python API 修改 + Python legacy test
- 构建要求：无需重新编译 C++ core；运行时需加载与 checkout 对应的 Python 源码。

## 7. 风险自查

- 泄露风险：`instruction.md` 仅描述公开行为和验收标准，不暴露具体实现行。
- 环境风险：低；需要保证测试运行时加载目标 checkout 的 Python 实现。
- flaky 风险：低；断言只检查确定性的 shape 与 `stop_gradient` 状态。
