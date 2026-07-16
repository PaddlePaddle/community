# Task Proposal: PaddlePaddle__Paddle-78238

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78238`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78238
- PR 标题：`[ZeroDim] fix put_along_axis with 0-size indices tensor`
- `base_commit`：`ae907b878e91dbabf3582da99f8b05a46b588fc2`
- merged 时间：`2026-03-20`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

完善 `paddle.put_along_axis` 和 `Tensor.put_along_axis_` 对零尺寸索引 Tensor 的支持。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle bug-fix PR，不是合成任务。
- **边界清楚**：目标行为集中在 `put_along_axis` 的 Python API 包装层，生产修改范围小。
- **可验证性**：修复前零尺寸索引用例稳定失败，修复后通过；非零尺寸索引用例可作为回归护栏。
- **环境稳定**：测试可在 CPU 环境完成，不依赖 GPU、分布式运行或外部服务。
- **行为导向**：测试只断言公开 API 的输出和原地语义，不检查实现源码，也不依赖特定修复写法。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[tensor, manipulation, put_along_axis, zero_size, broadcasting, python_api, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_put_along_axis_zero_size.py`
- P2P：非零尺寸索引的已有更新行为在修复前后均应通过。
- F2P 1：非原地接口接收零尺寸索引时，修复前应失败，修复后应返回与输入一致的结果。
- F2P 2：原地接口接收零尺寸索引时，修复前应失败，修复后应保持输入不变。
- 修复后预期：目标测试文件中的全部用例通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- patch 类型：Python-only
- Python 依赖：PaddlePaddle、NumPy、pytest
- 最小测试命令：`bash tests/test.sh`
- GPU、分布式运行、外部服务和额外数据集均不是必需项

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述目标行为和验收标准，不指定具体代码位置或实现方案。
- 环境风险：生产修改为 Python-only；verifier 可使用源码环境或等价的 Python overlay。
- flaky 风险：测试使用固定 shape 和确定性 shape/data 断言，不依赖随机数值结果。
- 拆分风险：原地与非原地接口属于同一零尺寸索引语义，适合作为一个任务验证。
