# Task Proposal: PaddlePaddle__Paddle-41202

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-41202`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/41202
- PR 标题：`Add AutoTune to reader.py for DataLoader`
- `base_commit`：`23d1b3e8ed8187bfb3bd926934dd6cc71e691e53`
- merged 时间：`2022-04-22T04:31:39Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

DataLoader 缺少按实际读取耗时自动选择 worker 数量的能力，用户只能反复手动调整 `num_workers`。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：数据读取速度受数据处理逻辑和机器 CPU 数量影响，固定 worker 数量经常不是合适配置。
- **代表性**：任务覆盖配置开关、抽样数据集、普通与分布式 batch sampler，以及不同平台的兼容行为。
- **边界清楚**：只处理 DataLoader worker 数量的自动选择，不改变数据内容、batch 结构或训练计算。
- **非平凡性**：需要比较多个候选 worker 数量的读取开销，并正确处理抽样范围、提前停止和 sampler 重建。
- **环境友好性**：来源 PR 提供了只使用随机小数据集的 Python 单测，不依赖网络、真实模型或 GPU。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[dataloader, multiprocessing, performance_tuning, batch_sampler]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`python/paddle/fluid/tests/unittests/test_dataloader_autotune.py`
- 修复前预期：关闭 auto-tune 时的 P2P 保持通过；普通 DataLoader 和 DistributedBatchSampler 的启用场景均因 Base 不具备 auto-tune 支持而失败；完整测试脚本失败。
- 修复后预期：P2P、两个启用场景以及完整上游测试文件全部通过。
- P2P 候选：`TestAutoTune::test_dataloader_disable_autotune`。
- F2P 候选：`TestAutoTune::test_dataloader_use_autotune` 和 `TestAutoTune::test_distributer_batch_sampler_autotune`。
- 测试来源：`tests/test.patch` 基于 PR 合入时新增的测试覆盖做最小 benchmark 适配：将 Gold-only `set_autotune_config` 的导入延迟到两个 F2P 测试体内，并使 P2P 不依赖该新增 API，从而保证 Base 可收集完整角色节点；其余测试逻辑与上游覆盖保持一致。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用可导入的 CPU Paddle wheel 作为运行载体；对于历史 `paddle.fluid.reader` 与当前 wheel 的模块差异，由 verifier 从 checkout 源码提取目标函数和类并执行真实控制流。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述用户需要的自动选择行为和兼容要求，不说明 Gold patch 的类名、循环结构或内部阈值。
- 环境风险：PR 较早，当前 Paddle wheel 可能不再提供 `paddle.fluid.reader`；cross script 使用受控 compatibility overlay，不覆盖已安装 Paddle 文件。
- flaky 风险：上游测试使用小型内存数据集；verifier 限制搜索用 CPU 数量，避免随机器核数扩大测试范围。
- 拆分风险：低。production 修改集中在 DataLoader worker 自动选择这一项功能。
