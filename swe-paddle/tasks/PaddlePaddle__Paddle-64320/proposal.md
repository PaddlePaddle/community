# Task Proposal: PaddlePaddle__Paddle-64320

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-64320`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/64320
- PR 标题：`【Hackathon 6th No.17】为 Paddle 新增 sparse.mask_as API -part`
- `base_commit`：`605f5e20305db0e4932a20d3e0e6cf7d7d9631d8`
- merged 时间：`2024-06-07T09:10:27Z`
- 你的身份：原 PR 作者
- 后续联系人：megemini

## 2. 问题一句话

为 Paddle 新增 `sparse.mask_as` API，支持根据给定的稀疏 mask 从稠密 Tensor 中提取对应位置的值，输出稀疏 Tensor。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 6th 框架开发 PR，关联 RFC（community#901），不是合成任务。
- **代表性**：它覆盖稀疏 CPU/GPU kernel 开发、CSR/COO 两种稀疏格式、Python API 封装、autograd 反向注册、以及 YAML op 定义，是典型的完整稀疏算子开发流程。
- **边界清楚**：目标行为集中在 `sparse.mask_as` API 的正确实现，测试补丁可以直接暴露目标行为。CSR 格式仅支持 2-D 和 3-D，边界清晰。
- **非平凡性**：该任务涉及 C++ kernel 实现（含 CSR 2D/3D 两种索引计算）、GPU CUDA kernel、反向梯度、Python API 封装和 YAML op 注册，不是纯 Python 或纯配置修改。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu` / `cuda`
- 设备范围：`single_gpu`
- 模块标签：`[sparse, python_api, cpu_kernel, gpu_kernel, autograd, yaml_op]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_sparse_mask_as_op.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，`test_sparse_mask_as_op.py` 中新增的 `mask_as` 相关测试应 fail（API 不存在或 kernel 未实现）。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：`test_sparse_mask_as_op.py` 为 PR 新增文件，无存量测试。建议从同模块存量稀疏测试中选取回归护栏，例如 `test_sparse_utils_op.py`、`test_sparse_unary_op.py` 等，可由 verifier 自动抽取稳定 nodeid。

## 6. 环境与资源

- 资源需求：CPU + GPU（CUDA kernel 涉及 GPU 编译）
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无，建议后续补充 source-build Dockerfile
- patch 类型：含 C++ CPU kernel + CUDA GPU kernel + Python API + YAML op 定义
- 环境建议：该样本涉及 C++ 和 CUDA kernel，需要 source build
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为和验收标准，不直接指出具体修改行。
- 环境风险：该样本涉及 C++ 和 CUDA kernel，历史 commit 复现可能需要 source build。
- flaky 风险：需要 verifier 重复运行目标测试，并抽取稳定 F2P/P2P nodeid。
- 拆分风险：该 PR 的目标集中在新增 `sparse.mask_as` API，适合作为一个样本。
