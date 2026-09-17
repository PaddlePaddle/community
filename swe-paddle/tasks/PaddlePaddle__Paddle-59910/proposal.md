# Task Proposal: PaddlePaddle__Paddle-59910

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-59910`
- Issue 链接：https://github.com/PaddlePaddle/Paddle/issues/58067
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/59910
- PR 标题：`【PIR API adaptor No.31】Migrate paddle.distribution.Normal into pir`
- `base_commit`：`5b3bc5043bcbc919a4bb9e69f421d6001814b365`
- merged 时间：`2024-01-16T03:31:00Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

让 `paddle.distribution.Normal` 的 Python / NumPy 参数在 PIR static graph 中遵循正确的参数转换与 broadcast 语义。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入 PaddlePaddle PR #59910，并对应 PIR Python API 适配工作。
- **代表性**：覆盖 Python API 与 PIR static graph 的边界行为，包括参数转换与 broadcast。
- **边界清楚**：production change 集中在 `python/paddle/distribution/distribution.py`；原 PR 同时修改了 `test/distribution/test_distribution_normal.py`。
- **非平凡性**：需要理解不同执行模式下 Tensor/PIR Value 的构造约束以及参数 broadcast，而不是简单参数检查。
- **环境友好性**：production patch 为 Python-only；独立测试直接执行 checkout 中 `_to_tensor` 的真实控制流，不需要 source build、GPU、网络或外部数据。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[pir, distribution, normal, static_graph, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr59910_normal_pir.py`
- 修复前预期：legacy-mode P2P 通过；两个 PIR-mode F2P 因仍走 legacy Tensor 创建路径而失败；完整 `tests/test.sh` 失败。
- 修复后预期：P2P 与两个 PIR-mode F2P 全部通过，完整目标测试文件与 `tests/test.sh` 均通过。
- P2P 候选：`test_legacy_mode_preserves_numpy_broadcast_conversion`。
- F2P 候选：`test_pir_mode_converts_numpy_parameters_without_legacy_tensor_creation`、`test_pir_mode_preserves_broadcasted_parameter_shapes_and_values`。

独立测试采用 SWE-Paddle 已使用的 `test/swe_paddle/` contract-test 形式，并通过 AST 从 checkout production file 提取 `Distribution._to_tensor` 执行真实控制流。这样避免历史 `test/distribution` 上游测试对 Paddle build-tree import path 的依赖，同时不检查源码字符串、不依赖具体局部变量名。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用可导入 Paddle、pytest、NumPy 的 CPU Python 环境；测试本身不覆盖或修改已安装 Paddle 包。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述 observable behavior，不给出 Gold patch 的具体实现步骤。
- 环境风险：避免直接运行依赖 build-tree import layout 的历史 `test/distribution` 文件；测试从仓库根目录运行，不需要自定义 `PYTHONPATH`。
- flaky 风险：测试不使用随机采样、网络、多进程、GPU 或 timing；NumPy 输入固定。
- 拆分风险：Gold production scope 仅一个 Python 文件；PR 的上游 test 修改仅作为来源证据，不并入 solution patch。
