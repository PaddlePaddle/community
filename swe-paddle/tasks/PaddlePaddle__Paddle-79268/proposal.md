# Task Proposal: PaddlePaddle__Paddle-79268

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-79268`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79268
* PR 标题：`[API Compatibility] Add alias paddle.utils.data.DistributedSampler for paddle.io.DistributedBatchSampler`
* `base_commit`：`722421e3a49eadf5ea774639c3d8147aced333ce`
* merged 时间：`2026-06-08`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

为数据加载模块新增公开的 `DistributedSampler` 兼容入口，并允许 `DistributedBatchSampler` 通过显式 seed 控制不同 epoch 下的 shuffle 顺序。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：任务来自已合入的 Paddle API Compatibility PR，不是人工构造的需求。
* **类型多样性**：该任务同时涉及公共 API 新增和现有接口的参数扩展，可补充 SWE-Paddle 中以 bug fix 为主的任务类型。
* **可观察性强**：公共接口是否正确导出、构造参数是否生效以及迭代顺序是否符合预期，都可以通过稳定的运行时行为验证。
* **回归边界清楚**：`DistributedBatchSampler` 在关闭 shuffle 时的分片、批次划分和长度计算可以作为 P2P，验证现有行为未受到影响。
* **非平凡性**：任务既要保持 `DistributedBatchSampler` 原有构造方式的兼容性，又要正确处理 seed 与 epoch 对 shuffle 的共同影响，并提供新的公共兼容入口。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[data_loader, distributed_sampler, api_compatibility, seed, public_api, python_only]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr79268_distributed_sampler.py`
* 修复前预期：现有 `DistributedBatchSampler` 的 P2P 测试应通过；显式 seed 和公开 `DistributedSampler` 相关的 F2P 测试应失败。
* 修复后预期：应用 production-only `solution/code.patch` 后，P2P 与全部 F2P 均应通过。
* P2P：固定 dataset、rank、`num_replicas` 且 `shuffle=False` 时，现有分片结果、批次划分和 `__len__` 行为保持不变。
* F2P 1：`DistributedBatchSampler(seed=...)` 能够接收显式 seed；在其他配置相同的情况下，相同 seed 和 epoch 产生相同顺序，不同 seed 能够产生不同的 shuffle 顺序。
* F2P 2：`paddle.utils.data.DistributedSampler` 被正确公开导出，支持通过位置参数和关键字参数传入 dataset、`num_replicas`、rank、shuffle、seed 和 `drop_last`，并表现出预期的分布式采样行为。

## 6. 环境与资源

* 资源需求：CPU
* GPU 是否必需：否
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python-only production patch
* Gold patch 边界：只包含 3 个 production 文件；原 PR 的 2 个 `test/legacy_test/` 文件不进入 `solution/code.patch`。
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：`instruction.md` 只描述公共接口、可配置 seed 和确定性 shuffle 等可观察行为，不透露具体修改位置、随机数计算方式或内部继承关系。
* 环境风险：通过 AST overlay 隔离 source checkout 与当前运行环境之间的版本差异，避免依赖历史 Paddle package 的完整导入和 native extension。
* flaky 风险：测试固定 seed、epoch、dataset 长度、rank 和 `num_replicas`，并直接比较采样结果，不依赖随机命中、并发或异步时序。
* distributed 风险：测试显式传入 `num_replicas` 和 rank，不依赖真实进程组、分布式启动器或环境变量。
* 拆分风险：公开 `DistributedSampler` 入口和 `DistributedBatchSampler` 的 seed 参数共同构成该 PR 的分布式采样兼容性改进，适合作为一个完整任务。
