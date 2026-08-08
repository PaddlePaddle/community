# Task Proposal: PaddlePaddle__Paddle-64519

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-64519`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/64519
- PR 标题：`【Hackathon 6th No.2】【Typing】为 Paddle 新增 cholesky_inverse API -part`
- `base_commit`：`2d746f9719ddd35e9e9f1330b019d996bdafbfac`
- merge commit：`3dcee14dd39d17c8d218624a5cb6a4ea437c217b`
- merged 时间：`2024-06-11T02:33:18Z`
- 后续联系人：megemini

## 2. 问题一句话

为 Paddle 新增 `cholesky_inverse` API，根据对称正定矩阵的 Cholesky 因子（下三角或上三角）计算其逆矩阵。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 6th 框架开发 PR，关联 RFC（community#896），不是合成任务。
- **代表性**：它覆盖 Python 层线性代数 API 的完整落地链路：`paddle.linalg` / `paddle.tensor` 命名空间导出、tensor 方法注册、类型注解（`from __future__ import annotations`）、输入校验与数学实现。
- **边界清楚**：目标行为集中在 `cholesky_inverse` 的正确实现与输入校验（2-D 方阵、upper 分支），测试补丁可直接暴露目标行为。
- **非平凡性**：需要理解 Cholesky 分解的三角结构（`UU^T` vs `U^TU`）与既有 `paddle.linalg.inv` 的组合，不是纯配置修改。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu`
- 模块标签：`[python_api, linalg, typing]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_linalg_cholesky_inverse.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`test_linalg_cholesky_inverse.py` 中 `cholesky_inverse` 相关测试应 fail（API 不存在）。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：`test_linalg_cholesky_inverse.py` 为 PR 新增文件，无存量测试；可从同模块存量线性代数测试（如 `test_linalg_solve.py`、`test_cholesky_op.py` 等）中选取回归护栏。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无，建议后续补充 source-build Dockerfile
- patch 类型：纯 Python（linalg API 封装，基于既有 `paddle.linalg.inv` 实现）
- 环境建议：该样本仅涉及 Python 层改动，source build 后运行 Python 测试即可
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为和验收标准，不直接指出具体修改行。
- 环境风险：仅 Python 改动，复现成本低于含 C++ kernel 的样本。
- flaky 风险：`cholesky_inverse` 为确定性数值计算，flaky 风险低；verifier 仍应重复运行抽取稳定 F2P/P2P nodeid。
- 拆分风险：该 PR 的目标集中在新增 `cholesky_inverse` API，适合作为一个样本。
