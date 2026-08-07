# Task Proposal: PaddlePaddle__Paddle-57827

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-57827`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/57827
- PR 标题：`【PIR】fused_elemwise_add_activation`
- `base_commit`：`8b1a29ba9bafc16116f97422574e85d208540332`（#57827 合入的父提交）
- gold commit：`3ac5e693b34eb3164fe076d489dc01bea9170843`
- merged 时间：`2023-11-15`
- 你的身份：原 PR 作者（GitHub @yangguohao）
- 后续联系人：GitHub @yangguohao

## 2. 问题一句话

在 PIR 路径下补齐 `fused_elemwise_add_activation` 算子支持（含前向 / 反向与 infermeta），使相关动转静 / build strategy 用例可在 PIR 下正确运行。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 PIR 算子迁移 PR，不是合成任务。
- **代表性**：覆盖 fusion infermeta、PIR op YAML、translator / adaptor 与兼容配置，属于典型的融合算子迁 PIR 样本。
- **边界清楚**：目标是该 fused op 在 PIR 下可用且语义正确；不扩展到其他无关 fusion 算子。
- **非平凡性**：需要同时补齐元信息、图翻译 / 适配与 PIR 注册；只打开测试开关不够。
- **区分度潜力**：漏掉 backward、infermeta 或 adaptor 任一侧，都会被完整验收拦住。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[pir, fusion, fused_elemwise_add_activation, infermeta, yaml_config]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/dygraph_to_static/test_build_strategy.py` 中与该 fused op / PIR 相关的用例
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，PIR 下相关 F2P 用例 fail 或 error。
- 修复后预期：继续应用 `solution/code.patch` 并重新构建后，F2P 与相关 P2P 均通过。
- P2P 护栏：同文件中既有无关 build strategy 用例继续通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- 是否需要 GPU：否
- patch 类型：含 C++ / YAML / infermeta，**需 source build**
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只写问题、复现、期望行为与验收标准；不包含衍生 PR 链接、diff、具体修改文件、具体实现步骤或答案路径。
- 环境风险：融合算子 + PIR 路径需要 source build；历史 commit 依赖需固定。
- flaky 风险：低到中；动转静策略相关用例需确认稳定 nodeid。
- 拆分风险：该 PR 目标集中在单一 fused op 的 PIR 支持，适合作为一个独立样本。
