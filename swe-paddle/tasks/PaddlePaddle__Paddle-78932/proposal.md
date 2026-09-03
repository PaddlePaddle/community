# Task Proposal: PaddlePaddle__Paddle-78932

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78932`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78932
- PR 标题：`[API Compatibility] Support vararg and add alias for paddle.io.TensorDataset`
- `base_commit`：`7b7e53fd28956700e5ed1ce68eb2aaeb59829777`
- merged 时间：`2026-05-11T08:51:45Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

`TensorDataset` 无法直接接收一个或多个 Tensor 作为独立参数，并且不能从 `paddle.utils.data` 使用同名 API。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：目标调用方式在数据加载代码迁移时很常见，当前会直接报参数错误或得到错误的数据集长度。
- **代表性**：要求同时兼容 list/tuple、单 Tensor 和多 Tensor 三种入参形式，与常见的 Python API 兼容问题一致。
- **边界清楚**：修改集中在 `TensorDataset` 的参数处理和公开导出，不涉及 DataLoader worker 或数据集外部资源。
- **非平凡性**：单 Tensor 在 Python 中可迭代，不能简单当作 Tensor 列表处理；还要保证共享参数适配逻辑不破坏已有 API。
- **环境友好性**：所有目标测试只使用小型 CPU Tensor，可稳定重复。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[data, dataset, api-compatibility]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/ai_edited_test/test_ai_dataloader.py` 和 `test/legacy_test/test_paddle_utils_data.py`
- 修复前预期：现有 list 调用正常；多 Tensor 位置参数报错，单 Tensor 被误解为可迭代的 Tensor 集合，`paddle.utils.data.TensorDataset` 不可用。
- 修复后预期：旧 list 调用保持不变，单 Tensor 和多 Tensor 调用返回正确长度与 tuple item，新公开入口可正常访问。
- P2P 候选：`test_tensor_dataset_basic`、`test_tensor_dataset_1d`、`test_tensor_dataset_iter`。
- F2P 候选：`test_tensor_dataset_varargs`、`test_tensor_dataset_varargs_single`、`TestAlias::test_compatibility`。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务描述只说明用户可见的入参方式和数据集行为，未说明 Gold patch 的具体实现。
- 环境风险：历史 checkout 与已安装 wheel 可能有 Python 模块差异，cross verifier 需要使用受控的 source overlay。
- flaky 风险：测试只校验 shape、长度、tuple 结构和 API 可访问性，不依赖随机数值、worker 时序或外部资源。
- 拆分风险：varargs 和 `paddle.utils.data` 导出都围绕同一个 `TensorDataset` API 兼容问题，与来源 PR 的单一目标一致。
