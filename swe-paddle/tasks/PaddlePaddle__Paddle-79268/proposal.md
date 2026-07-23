# Task Proposal: PaddlePaddle__Paddle-79268

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79268`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79268
- PR 标题：`[API Compatibility] Add alias paddle.utils.data.DistributedSampler for paddle.io.DistributedBatchSampler`
- `base_commit`：`722421e3a49eadf5ea774639c3d8147aced333ce`
- merged 时间：`2026-06-08`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为数据加载模块增加公开的 `DistributedSampler` 兼容入口，并允许 `DistributedBatchSampler` 使用显式 seed 控制跨实例、跨 epoch 的 shuffle 顺序。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle API Compatibility PR，不是合成需求。
- **类型多样性**：这是公共 API addition 与参数能力增强，而不是 bug fix。
- **可观察性强**：公开导出、构造参数和迭代顺序都能通过稳定的运行期行为验证。
- **回归边界清楚**：现有无 shuffle 的 `DistributedBatchSampler` 分片与 batch 行为可作为 P2P。
- **非平凡性**：需要同时处理旧 sampler 的参数兼容、随机种子与 epoch 组合，以及新公共入口的导出和参数转发。
- **环境友好性**：单进程 CPU 即可构造固定 rank/replica 的 sampler，无需真实 distributed launch、GPU 或外部数据集。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[data_loader, distributed_sampler, api_compatibility, seed, public_api, python_only]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr79268_distributed_sampler.py`
- 修复前预期：现有 `DistributedBatchSampler` P2P 通过；显式 seed 和公开 `DistributedSampler` 两个 F2P 失败。
- 修复后预期：应用 production-only `solution/code.patch` 后，全部目标测试通过。
- P2P：固定 dataset、rank、replica 和 `shuffle=False` 时，既有分片、batch 和 `__len__` 行为保持不变。
- F2P 1：`DistributedBatchSampler(seed=...)` 接受显式 seed；相同 `seed + epoch` 产生相同顺序，不同 seed 产生不同顺序。
- F2P 2：`paddle.utils.data.DistributedSampler` 被公开导出，并将 dataset、replica、rank、shuffle、seed 与 drop-last 参数转发到对应的 distributed batch sampler 行为。

## 6. 环境与资源

- 资源需求：CPU
- GPU 是否必需：否
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production patch
- 环境建议：从 checkout 中提取并执行真实 `DistributedBatchSampler`、`DistributedSampler` 类与 `paddle.utils.data` 导出语句，在 controlled module namespace 中验证行为。
- Gold patch 边界：只包含 3 个 production 文件；原 PR 的 2 个 `test/legacy_test/` 文件不进入 `solution/code.patch`。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述公共 API 与确定性行为，不给出具体修改行或补丁实现。
- 环境风险：AST overlay 避免导入历史 checkout 的完整 Paddle package 和 native extension。
- flaky 风险：随机顺序由固定 seed、epoch、dataset 长度、rank 和 replica 数完全决定。
- distributed 风险：测试显式传入 `num_replicas` 与 `rank`，不依赖真实进程组或环境变量。
- 拆分风险：公开 alias 与 seed 参数共同构成该 PR 的单一 distributed sampler 兼容性目标。
