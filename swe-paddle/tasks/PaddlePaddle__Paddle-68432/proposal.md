# Task Proposal: PaddlePaddle__Paddle-68432

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-68432`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/68432
- PR 标题：`【Hackathon 7th No.18】为稀疏计算添加复数支持2 -part`
- `base_commit`：`979489bc3280e682f2ce8996d9b0e154ec425a59`
- merged 时间：`2024-09-29T11:58:34Z`
- 你的身份：原 PR 作者 / reviewer / 熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为 Paddle 稀疏计算中的 `multiply_coo_coo`、`multiply_csr_csr`、`divide_coo_coo`、`divide_csr_csr` 添加复数支持。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 7th 框架开发 PR，不是合成任务。
- **代表性**：它覆盖稀疏 CPU kernel、复数 dtype 支持、Python sparse API 行为，以及复数 autograd 公式。
- **边界清楚**：目标行为集中在稀疏 elementwise multiply / divide 对复数输入的支持，测试补丁可以直接暴露目标行为。
- **非平凡性**：这个任务不只是注册复数 dtype。正确修复还需要在复数乘除梯度中处理 conjugation，否则数值会出现明显误差。
- **区分度信号**：在 CPU10 pilot 中，Claude Opus 4.8 通过该样本，ERNIE 5.1 未通过，说明它能拉开模型在框架级细节上的差异。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[sparse, complex, autograd, cpu_kernel, python_api, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_sparse_elementwise_op.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，复数稀疏 elementwise 相关测试应 fail / error。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：同文件中已有的 sparse elementwise 非复数用例，可由 verifier 自动抽取稳定 nodeid 作为回归护栏。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无，建议后续补充 source-build Dockerfile
- patch 类型：含 C++ kernel 修改 + Python API 修改
- 环境建议：该样本涉及 sparse CPU kernel，不能只依赖已有 wheel；需要 source build 或等价的已编译环境
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为和验收标准，不直接指出具体修改行。
- 环境风险：该样本涉及 C++ kernel，历史 commit 复现可能需要 source build。
- flaky 风险：需要 verifier 重复运行目标测试，并抽取稳定 F2P/P2P nodeid。
- 拆分风险：该 PR 的目标集中在稀疏复数 elementwise 支持，适合作为一个样本。
