# Task Proposal: PaddlePaddle__Paddle-59021

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-59021`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/59021
- PR 标题：`【PIR】fix test_len`
- `base_commit`：`a53f40972d9dea85b44e6eae288f14c1bd01e3a7`（#59021 合入的父提交）
- gold commit：`3af9eb7eb21f80e81f3573c427feeebbd621a72a`
- merged 时间：`2023-11-27`
- 你的身份：原 PR 作者（GitHub @yangguohao）
- 后续联系人：GitHub @yangguohao

## 2. 问题一句话

修复 PIR 下 `test_len` 中 `len_with_selected_rows` 相关失败，并开放 PIR 路径上对应 fused elemwise add activation 相关测试，使相关用例可在 PIR 下通过。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 PIR 兼容修复 PR，针对真实单测失败，不是合成任务。
- **代表性**：覆盖 PIR adaptor / 算子工具路径与测试门禁开启，属于 PIR 迁移中的兼容性修复样本。
- **边界清楚**：目标是 `test_len` 相关 PIR 用例通过，并正确开启既定 PIR 测试；不扩展到无关 len / SelectedRows API 重设计。
- **非平凡性**：需要定位 PIR 执行链路上的适配缺口，而不是简单跳过测试。
- **区分度潜力**：只改 CMake 白名单而不修运行时适配，或只改一处绑定，都会被完整验收拦住。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[pir, selected_rows, test_len, adaptor, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/` 下与 `test_len` / PIR 开启相关的用例（以本 PR 打开的测试目标为准）
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，PIR 下 `len_with_selected_rows` 相关 F2P 用例 fail 或 error。
- 修复后预期：继续应用 `solution/code.patch` 并重新构建后，F2P 与相关 P2P 均通过。
- P2P 护栏：同模块中既有非目标 len / 无关用例继续通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- 是否需要 GPU：否
- patch 类型：含 C++ / YAML / 测试门禁改动，**需 source build**
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只写问题、复现、期望行为与验收标准；不包含衍生 PR 链接、diff、具体修改文件、具体实现步骤或答案路径。
- 环境风险：历史 PIR 路径依赖 source build；测试门禁变更需与 verifier 的用例选取一致。
- flaky 风险：低；以确定性单测为主。
- 拆分风险：该 PR 变更面较小且围绕同一 PIR 兼容目标，适合作为一个独立样本。
