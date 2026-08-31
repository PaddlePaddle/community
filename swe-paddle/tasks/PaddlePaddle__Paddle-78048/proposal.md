# Task Proposal: PaddlePaddle__Paddle-78048

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78048`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78048
- PR 标题：`[API Compatibility No.62、73、234] Add parameter alias support for dsplit、hsplit、vsplit - part`
- `base_commit`：`3f270c40db7776481d69176ee09222b3437d92bb`
- merged 时间：`2026-03-05T10:11:27+08:00`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

`paddle.hsplit`、`paddle.dsplit` 和 `paddle.vsplit` 无法使用 `input`、`indices` 或 `sections` 这些常见参数名调用，与已有代码的兼容性不足。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自真实的 Paddle API 兼容性改进，用户在迁移或复用切分代码时可直接遇到。
- **代表性**：要求在多个相关公开 API 中统一处理参数别名，同时保留原有调用方式。
- **边界清楚**：生产修改集中在单个 Python 文件，目标行为可由三组 API compatibility 测试直接验证。
- **非平凡性**：修复不能只针对某一个函数或某一个参数特判，还要保证新旧参数名返回一致的切分结果。
- **环境友好性**：可在 CPU 环境使用小型 Tensor 稳定复现，不依赖外部资源。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[tensor, split-api, api-compatibility]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_api_compatibility.py`
- 修复前预期：`input`、`indices` 和 `sections` 别名调用失败，目标测试不通过。
- 修复后预期：三个 API 的原参数名、新别名和 Tensor method 调用均返回与 NumPy 参考结果一致的切分结果。
- P2P 候选：位置参数调用、`x`/`num_or_indices` 关键字参数调用以及已有 Tensor method 调用。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务描述只说明对外行为和兼容性要求，未给出 Gold patch 的具体修改方式。
- 环境风险：历史 checkout 与已安装 wheel 可能存在 Python 层差异，因此需要按 environment 说明加载 checkout 中的目标模块。
- flaky 风险：测试使用固定随机种子和小型 CPU Tensor，不依赖时序、网络或外部数据。
- 拆分风险：三个 API 共享同一类参数别名兼容问题，且修改位于同一生产文件，无需拆成多个任务。
