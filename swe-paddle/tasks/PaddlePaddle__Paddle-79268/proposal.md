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

为 `paddle.io.DistributedBatchSampler` 增加 `paddle.utils.data.DistributedSampler` API 别名，并增加可配置的 `seed` 参数。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：任务来自已合入的 Paddle API Compatibility PR，不是人工构造的需求。
* **类型多样性**：该任务同时涉及 API 别名新增和现有接口参数扩展，可补充 SWE-Paddle 中以 bug fix 为主的任务类型。
* **可观察性强**：可以直接验证 `DistributedSampler` 是否可用、构造参数是否生效，以及 `seed` 是否正确影响 shuffle 顺序。
* **回归边界清楚**：`DistributedBatchSampler` 原有的构造方式、数据分片、批次划分和长度计算均可作为 P2P。

## 4. 任务类型和标签

* 任务类型：`feature_enhancement`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[data_loader, distributed_sampler, api_compatibility, seed, public_api, python_only]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/swe_paddle/test_pr79268_distributed_sampler.py`
* 修复前预期：现有 `DistributedBatchSampler` 的 P2P 测试应通过；`seed` 参数和 `DistributedSampler` API 相关的 F2P 测试应失败。
* 修复后预期：应用 production-only `solution/code.patch` 后，P2P 与全部 F2P 均应通过。
* P2P：固定 dataset、rank、`num_replicas` 且 `shuffle=False` 时，现有分片结果、批次划分和 `__len__` 行为保持不变。
* F2P 1：`DistributedBatchSampler` 支持显式传入 `seed`；其他配置相同时，相同 seed 和 epoch 产生相同顺序，不同 seed 能够产生不同的 shuffle 顺序。
* F2P 2：`paddle.utils.data.DistributedSampler` 可以正常访问，并支持通过位置参数和关键字参数传入 dataset、`num_replicas`、rank、shuffle、seed 和 `drop_last`。

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

* 泄露风险：`instruction.md` 只描述 `DistributedSampler` API、`seed` 参数和 shuffle 行为，不透露具体修改位置或实现步骤。
* 环境风险：通过 AST overlay 隔离 source checkout 与当前运行环境之间的版本差异，避免依赖历史 Paddle package 的完整导入和 native extension。
* flaky 风险：测试固定 seed、epoch、dataset 长度、rank 和 `num_replicas`，并直接比较采样结果，不依赖随机命中、并发或异步时序。
* distributed 风险：测试显式传入 `num_replicas` 和 rank，不依赖真实进程组、分布式启动器或环境变量。
* 拆分风险：`DistributedSampler` API 别名和 `DistributedBatchSampler` 的 `seed` 参数属于同一个 API compatibility PR，适合作为一个完整任务。
