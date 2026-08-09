# Task Proposal: PaddlePaddle__Paddle-58917

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-58917`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/58917
- PR 标题：`【Hackathon 5th No.32】为 Paddle 新增 tensor_split / hsplit / dsplit API -part`
- `base_commit`：`46e3dfeaa50ec97edeebb1acd5205f5cd702bf5c`
- merge commit：`538905c80c938a0f96504ae3983baa3d29be8b9a`
- merged 时间：`2023-12-13T03:42:02Z`
- 后续联系人：megemini

## 2. 问题一句话

为 Paddle 新增 `tensor_split` / `hsplit` / `dsplit` API，支持按份数（可不均分）或按索引列表沿指定轴切分张量。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 5th 框架开发 PR，关联 RFC（community#682），不是合成任务。
- **代表性**：它覆盖 Python 层张量切分 API 的完整落地链路：顶层命名空间与 `paddle.tensor` 命名空间导出、tensor 方法绑定、整数切分（base/mod 非等分逻辑）与索引切分（含负索引处理）两条分支、既有 `vsplit` 的重构复用。
- **边界清楚**：目标行为集中在三种切分语义（等分/非等分、索引切分、axis 包装），测试补丁可直接暴露目标行为。
- **非平凡性**：需要正确处理非等分切分的尺寸分配、负索引、负 axis、非法输入校验，并保持 `vsplit` 行为不变；非纯配置修改。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu`
- 模块标签：`[python_api, tensor, manipulation]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_splits_api.py`（PR 重写并涵盖了原有 vsplit 测试）
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`test_splits_api.py` 中 `tensor_split` / `hsplit` / `dsplit` 相关测试应 fail（API 不存在）。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：`test_splits_api.py` 为 PR 重写文件（涵盖原 vsplit 测试），可整体作为回归护栏。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无，建议后续补充 source-build Dockerfile
- patch 类型：纯 Python（基于既有 `split` / `slice` 原语实现）
- 环境建议：该样本仅涉及 Python 层改动，source build 后运行 Python 测试即可
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为和验收标准，不直接指出具体修改行。
- 环境风险：仅 Python 改动，复现成本低于含 C++ kernel 的样本。
- flaky 风险：`tensor_split` 系列为确定性张量操作，flaky 风险低；verifier 仍应重复运行抽取稳定 F2P/P2P nodeid。
- 拆分风险：该 PR 的目标集中在新增三个切分 API，适合作为一个样本。
