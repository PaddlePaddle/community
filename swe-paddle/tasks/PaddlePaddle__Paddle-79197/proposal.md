# Task Proposal: PaddlePaddle__Paddle-79197

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79197`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79197
- PR 标题：`[API Compatibility] Support param optimizer for lr_scheduler`
- `base_commit`：`06d8af53d39ef6622689bab27e1cd03a2ffab0f3`
- merged 时间：`2026-06-08T06:44:24Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

常用 learning-rate scheduler 只能接收数值学习率，不能直接接收已经创建好的 optimizer，导致兼容写法报错且无法自动关联二者。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：训练代码迁移时，先创建 optimizer、再把 optimizer 交给 scheduler 是常见写法。
- **代表性**：任务涉及多个 scheduler 的一致参数行为，以及 scheduler 与 optimizer 的状态关联。
- **边界清楚**：只调整 learning-rate scheduler 的初始化兼容性，不修改优化算法、算子或训练结果计算。
- **非平凡性**：需要同时支持位置参数和关键字参数，处理冲突输入，并确保各 scheduler 的学习率曲线不变。
- **环境友好性**：来源 PR 单测使用小型 CPU 网络和本地随机输入，不依赖 GPU、外部数据或网络。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[optimizer, lr-scheduler, api-compatibility]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_lr_scheduler.py`
- 修复前预期：新增的 optimizer 参数测试失败；已有 `learning_rate` 调用测试通过。
- 修复后预期：optimizer 可通过位置参数或关键字参数传入，scheduler 使用 optimizer 当前学习率并被 optimizer 持有，PR 新增的六个目标用例与已有 regression case 全部通过。
- P2P 候选：`TestCosineAnnealingWarmRestarts::test_CosineRestartsLR`
- F2P 候选：`test_exponential_decay`、`test_cosine_annealing_decay`、`test_cosine_annealing_warm_restarts`、`test_multi_step_decay`、`test_reduce_on_plateau`、`test_step_decay`

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用兼容的已安装 Paddle wheel 承载运行时，由 verifier 覆盖 checkout 中两个精确 Python production files。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述开发者遇到的调用问题和期望行为，不说明 Gold patch 的具体实现方式。
- 环境风险：测试会进行小规模 CPU 前向、反向和 optimizer step，需要已安装 Paddle，但无需源码编译。
- flaky 风险：上游测试不使用网络、并发、外部数据集或随机时序，只验证确定的学习率序列和对象关联。
- 拆分风险：所有变更共同解决 scheduler 接收 optimizer 的兼容调用问题，没有混入其他功能。
