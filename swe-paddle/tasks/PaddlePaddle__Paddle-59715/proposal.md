# Task Proposal: PaddlePaddle__Paddle-59715

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-59715`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/59715
- PR 标题：`【Hackathon 5th No.36】 为 Paddle 新增 matrix_exp API -part`
- `base_commit`：`3edda65cca8d44a1bddea70ac6f04f2b95430e9c`
- merge commit：`db804cd018d1812e2c5629235b1f127a904e52c3`
- merged 时间：`2023-12-21T13:18:18Z`
- 后续联系人：megemini

## 2. 问题一句话

为 Paddle 新增 `matrix_exp` API，使用 scaling-and-squaring + Padé 近似计算方阵的矩阵指数。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 5th 框架开发 PR，关联 RFC（community#775），不是合成任务。
- **代表性**：它覆盖 Python 层线性代数数值算法的完整实现：scaling-and-squaring 流程、多阶 Padé 近似（3/5/7/9）、静态图/动态图双路径（`paddle.framework.in_dynamic_mode()` 分支）、命名空间导出与 tensor 方法注册。
- **边界清楚**：目标行为集中在 `matrix_exp` 的数值正确性（与参考实现一致），测试补丁可直接暴露目标行为。
- **非平凡性**：需要实现一个完整的数值算法（范数估计、缩放、Padé 近似、平方恢复），并保证静态图/动态图一致性，不是简单封装。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu`
- 模块标签：`[python_api, linalg, numerical]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_linalg_matrix_exp.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`test_linalg_matrix_exp.py` 中 `matrix_exp` 相关测试应 fail（API 不存在）。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：`test_linalg_matrix_exp.py` 为 PR 新增文件，无存量测试；可从同模块存量线性代数测试中选取回归护栏，由 verifier 自动抽取稳定 nodeid。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无，建议后续补充 source-build Dockerfile
- patch 类型：纯 Python（linalg 数值算法实现，基于既有 matmul/add/full 等原语）
- 环境建议：该样本仅涉及 Python 层改动，source build 后运行 Python 测试即可
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为和验收标准，不直接指出具体修改行。
- 环境风险：仅 Python 改动，复现成本低于含 C++ kernel 的样本。
- flaky 风险：`matrix_exp` 为确定性数值计算，flaky 风险低；verifier 仍应重复运行抽取稳定 F2P/P2P nodeid。
- 拆分风险：该 PR 的目标集中在新增 `matrix_exp` API，适合作为一个样本。
