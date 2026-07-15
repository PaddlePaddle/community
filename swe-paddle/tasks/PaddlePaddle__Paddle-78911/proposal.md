# Task Proposal: PaddlePaddle__Paddle-78911

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78911`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78911
- PR 标题：`fix recompute detection bug`
- `base_commit`：`6dfc086f8a3c2245ea1d75891386e82aa5721f15`
- merged 日期：`2026-05-09`
- 你的身份：contributor
- 后续联系人：TBD

## 2. 问题一句话

完善动态图 reentrant recompute 的上下文状态，使前向执行与反向重计算均可被正确识别。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle Bug Fix PR。
- **边界清楚**：生产修改集中在一个 Python 文件中的 recompute 执行路径。
- **验证稳定**：测试可在 CPU 动态图环境运行，不依赖 GPU、分布式多进程或外部服务。
- **行为可观测**：修复前后可直接比较前向与反向重计算期间的上下文状态。
- **规模适中**：补丁较小，但要求理解 PyLayer 反向重计算和上下文生命周期。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[recompute, autograd, dygraph, context_manager, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_recompute_context.py`
- 修复前预期：前向上下文及梯度回归用例通过；反向重计算上下文用例失败。
- 修复后预期：前向、反向重计算、上下文清理和梯度用例全部通过。
- P2P：验证 recompute 梯度结果及执行结束后的上下文清理。
- F2P：分别覆盖启用和关闭 RNG 状态保存的反向重计算路径。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at the base commit
- patch 类型：Python-only
- 最小测试命令：`bash tests/test.sh`
- Python-only 场景可使用源码安装或等价的 Python overlay 环境验证

## 7. 风险自查

- 泄露风险：任务描述仅陈述目标行为，不指定具体修改位置或实现方式。
- 环境风险：测试不依赖 GPU、NCCL 或真实分布式运行环境。
- flaky 风险：测试使用确定性的 Tensor 运算和显式状态断言。
- 拆分风险：目标集中在 reentrant recompute 上下文生命周期，适合作为单一任务。
