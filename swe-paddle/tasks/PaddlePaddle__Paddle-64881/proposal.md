# Task Proposal: PaddlePaddle__Paddle-64881

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-64881`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/64881
- PR 标题：`【Hackathon 6th No.8】NO.8 为 Paddle 新增 FeatureAlphaDropout API`
- `base_commit`：`d972f9ab8bb3d2ea5d1757a860ae45774e53b6eb`
- merge commit：`fb3154f4a8ece2a26ddc4438b9f684e4a32abe9e`
- merged 时间：`2024-06-28T04:04:39Z`
- 后续联系人：megemini

## 2. 问题一句话

为 Paddle 新增 `FeatureAlphaDropout` 层与 `feature_alpha_dropout` 函数式 API，按 channel 整体置零并保持 Alpha Dropout 的自归一化性质。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 6th 框架开发 PR，关联 RFC（community#913），不是合成任务。
- **代表性**：它覆盖 Python API 封装（函数式 + Layer 层）、既有实现的重构复用（`alpha_dropout` 抽出共享 `_feature_alpha_dropout_impl`）、以及静态图/动态图双路径测试，是典型的 Python 层 API 扩展任务。
- **边界清楚**：目标行为集中在 `feature_alpha_dropout` 的 channel 级 mask 语义（至少 2-D 输入、前两维共享 mask），测试补丁可直接暴露目标行为。
- **非平凡性**：需要理解既有 `alpha_dropout` 的 mask 生成与仿射缩放逻辑，并正确泛化为 feature 模式，同时保证原 API 行为不变；非纯配置修改。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu`
- 模块标签：`[python_api, nn, dropout]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_alpha_dropout.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`test_alpha_dropout.py` 中 `feature_alpha_dropout` / `FeatureAlphaDropout` 相关测试应 fail（API 不存在）。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：`test_alpha_dropout.py` 为 PR 新增文件，无存量测试；该文件同时覆盖既有 `alpha_dropout` 行为，可整体作为回归护栏。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无，建议后续补充 source-build Dockerfile
- patch 类型：纯 Python（nn functional API + Layer 封装），无 C++/CUDA kernel
- 环境建议：该样本仅涉及 Python 层改动，source build 后运行 Python 测试即可
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为和验收标准，不直接指出具体修改行。
- 环境风险：仅 Python 改动，复现成本低于含 C++ kernel 的样本。
- flaky 风险：`alpha_dropout` 相关测试依赖随机数，需 verifier 重复运行并抽取稳定 F2P/P2P nodeid。
- 拆分风险：该 PR 的目标集中在新增 `feature_alpha_dropout`，适合作为一个样本。
