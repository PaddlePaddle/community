# Task Proposal: PaddlePaddle__Paddle-59383

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-59383`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/59383 ；https://github.com/PaddlePaddle/Paddle/pull/60835 （后续小修，一并纳入）
- PR 标题：`【Hackathon No.4】为 Paddle 新增 masked_scatter API -part` / `fix masked_scatter`
- `base_commit`：`a8d5117371e8b9d16ff28011329bc04104eaf50a`（#59383 合入的父提交）
- gold endpoint：`a92999d0788ab7d4241a3daf9cadcb67566ef541`（#60835 合入后）
- merged 时间：`2023-12-13`（#59383）、`2024-01-17`（#60835）
- 你的身份：原 PR 作者（GitHub @yangguohao）
- 后续联系人：GitHub @yangguohao

## 2. 问题一句话

为 Paddle 新增 `masked_scatter` / `masked_scatter_` API（对齐常见 masked scatter 语义），并纳入后续小修以保证 mask 类型与静态图相关边界行为可用、可测。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 Hackathon No.4 新 API PR，以及合入后针对真实问题的小修，不是合成任务。
- **代表性**：覆盖 tensor manipulation 类 Python API、inplace 变体与单测契约，属于常见的 API 新增样本。
- **边界清楚**：目标是提供正确的 masked scatter 行为（含非 inplace / inplace），并保持既有 tensor API 不受影响；不扩展到无关 scatter / index 算子族。
- **非平凡性**：需要同时处理 API 暴露、masked 更新语义，以及后续暴露出的 mask 类型 / 静态图边界问题；只加空壳接口或只改测试不够。
- **区分度潜力**：漏掉 inplace、mask 约束或小修中的边界处理，都会被完整验收拦住。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, tensor_manipulation, masked_scatter, inplace, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_masked_scatter.py`，以及 `test/legacy_test/test_inplace.py` 中与 `masked_scatter` 相关的用例
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`masked_scatter` 相关 F2P 用例 fail 或 error（API 不存在或行为不正确）。
- 修复后预期：继续应用 `solution/code.patch` 后，F2P 与相关 P2P 均通过。
- P2P 护栏：同目录中既有 inplace / manipulation 无关用例继续通过，避免只为新 API 放宽断言。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- 是否需要 GPU：否
- patch 类型：纯 Python（#59383 新增 API + #60835 对实现的小修）
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只写问题、复现、期望行为与验收标准；不包含衍生 PR 链接、diff、具体修改文件、具体实现步骤或答案路径。
- 环境风险：历史 commit（2023-12）可用时代接近的 wheel 或 source checkout；纯 Python 改动通常无需 C++ 重建。
- flaky 风险：主路径应为确定性数值比较；若含随机初始化需固定 seed。
- 拆分风险：#60835 仅为 `masked_scatter` 小修，与主 PR 同属一个 API 目标，合并为一个样本更合理；gold 取相对 `base_commit` 的净效果。
