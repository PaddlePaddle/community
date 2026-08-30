# Task Proposal: PaddlePaddle__Paddle-58323

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-58323`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/58323
- PR 标题：`【Hackathon 5th No.33】为 Paddle 新增 atleast_1d / atleast_2d / atleast_3d API -part`
- `base_commit`：`431a0d53bd1578385ebcd3021f08d88ed6f75c70`
- merge commit：`557499bd9b511515309bcf8665b215ef7f016e71`
- merged 时间：`2023-11-16T12:29:50Z`
- 后续联系人：megemini

## 2. 问题一句话

为 Paddle 新增 `atleast_1d` / `atleast_2d` / `atleast_3d` API，将标量或低维输入转换为至少具有指定维度的张量。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 5th 框架开发 PR，关联 RFC（community#679），不是合成任务。
- **代表性**：它覆盖 Python 层张量操作 API 的完整落地链路：`paddle` 顶层命名空间与 `paddle.tensor` 命名空间导出、tensor 方法注册、可变参数处理（单输入返回 Tensor / 多输入返回列表）以及基于既有 reshape/unsqueeze 的实现。
- **边界清楚**：目标行为集中在三个 API 的补维语义（0-D/1-D/2-D 输入分别如何补维），测试补丁可直接暴露目标行为。
- **非平凡性**：需要正确处理可变参数、标量输入转换、不同维度的分支逻辑，并保证与 NumPy 语义一致；非纯配置修改。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu`
- 模块标签：`[python_api, tensor, manipulation]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_atleast_nd.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`test_atleast_nd.py` 中 `atleast_1d/2d/3d` 相关测试应 fail（API 不存在）。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：`test_atleast_nd.py` 为 PR 新增文件，无存量测试；可从同模块存量 manipulation 测试中选取回归护栏，由 verifier 自动抽取稳定 nodeid。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无，建议后续补充 source-build Dockerfile
- patch 类型：纯 Python（tensor manipulation API，基于既有 reshape/unsqueeze 实现）
- 环境建议：该样本仅涉及 Python 层改动，source build 后运行 Python 测试即可
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为和验收标准，不直接指出具体修改行。
- 环境风险：仅 Python 改动，复现成本低于含 C++ kernel 的样本。
- flaky 风险：`atleast_nd` 为确定性张量操作，flaky 风险低；verifier 仍应重复运行抽取稳定 F2P/P2P nodeid。
- 拆分风险：该 PR 的目标集中在新增三个 atleast API，适合作为一个样本。
