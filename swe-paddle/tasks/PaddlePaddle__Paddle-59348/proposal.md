# Task Proposal: PaddlePaddle__Paddle-59348

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-59348`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/59348
- PR 标题：`【PIR】add sequence_mask in pir`
- `base_commit`：`1001b3234973fb1fd2d6ede7afe918c82c792d66`（#59348 合入的父提交）
- gold commit：`669a3007e45b0b9f4600faa0a0ee3ff51fe90af3`
- merged 时间：`2023-12-08`
- 你的身份：原 PR 作者（GitHub @yangguohao）
- 后续联系人：GitHub @yangguohao

## 2. 问题一句话

在 PIR 路径下补齐 `sequence_mask` 算子支持，使相关动态图转静态 / sequence 用例可在 PIR 下正确运行并通过验证。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 PIR 迁移框架开发 PR，不是合成任务。
- **代表性**：覆盖算子 YAML / compat、infermeta、CPU/GPU kernel 与 PIR 注册，属于典型的「旧图算子迁 PIR」样本。
- **边界清楚**：目标是 `sequence_mask` 在 PIR 下可用且与既有语义一致；不扩展到其他 sequence 算子族。
- **非平凡性**：需要同时打通元信息、kernel 与 PIR 暴露路径；只改测试白名单不够。
- **区分度潜力**：漏掉 infermeta、kernel 或 PIR 注册任一侧，都会被完整验收拦住。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[pir, sequence_mask, infermeta, operator_kernel, yaml_config]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/sequence/` 下与 `sequence_mask` / PIR 开启相关的用例（以 `test/sequence/CMakeLists.txt` 变更覆盖的目标为准）
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，PIR 下 `sequence_mask` 相关 F2P 用例 fail 或 error。
- 修复后预期：继续应用 `solution/code.patch` 并重新构建后，F2P 与相关 P2P 均通过。
- P2P 护栏：同模块中既有非 PIR / 无关 sequence 用例继续通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- 是否需要 GPU：否（目标以 CPU 验证为主；PR 含 GPU kernel 改动）
- patch 类型：含 C++ / YAML / kernel，**需 source build**
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只写问题、复现、期望行为与验收标准；不包含衍生 PR 链接、diff、具体修改文件、具体实现步骤或答案路径。
- 环境风险：历史 PIR 迁移动态依赖 source build；GPU kernel 改动在 CPU-only 验证时可降级为不强制跑 GPU 用例。
- flaky 风险：主路径应为确定性数值比较；verifier 应抽取稳定 nodeid。
- 拆分风险：该 PR 目标集中在 `sequence_mask` 的 PIR 支持，适合作为一个独立样本。
