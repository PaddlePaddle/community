# Task Proposal: PaddlePaddle__Paddle-59847

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-59847`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/59847
- PR 标题：`【Hackathon 5th No.38】为 Paddle 新增 FractionalMaxPool2d / FractionalMaxPool3d API -kernel`
- `base_commit`：`600fc2f0e758d28c85c738c57ade718bef6daec5`
- merged 时间：`2024-01-12T08:55:21Z`
- 你的身份：原 PR 作者
- 后续联系人：megemini

## 2. 问题一句话

为 Paddle 新增 `nn.FractionalMaxPool2d` 和 `nn.FractionalMaxPool3d` API，支持分数阶最大池化操作。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle Hackathon 5th 框架开发 PR，关联 RFC（community#698 / #798），不是合成任务。
- **代表性**：它覆盖了 CPU/GPU kernel 开发、前向与反向梯度、infermeta、YAML op 注册、Python API 封装，以及完整的单测（op 级别 + API 级别），是典型的池化算子开发流程。
- **边界清楚**：目标行为集中在 `FractionalMaxPool2d` / `FractionalMaxPool3d` 的正确实现，包括前向计算、反向传播和 API 封装。验收标准由新增的四个测试文件覆盖。
- **非平凡性**：该任务涉及 C++ kernel 实现（含 CPU + GPU）、复杂的反向梯度公式、infermeta 推导、以及 Python API 的 functional 和 layer 两层封装，代码量较大（diff 约 4100 行），不是纯 Python 或简单配置修改。

## 4. 任务类型和标签

- 任务类型：`feature_implementation`
- 执行后端：`cpu` / `cuda`
- 设备范围：`single_gpu`
- 模块标签：`[python_api, cpu_kernel, gpu_kernel, autograd, yaml_op, pooling]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：
  - `test/legacy_test/test_fractional_max_pool2d_op.py`
  - `test/legacy_test/test_fractional_max_pool3d_op.py`
  - `test/legacy_test/test_fractional_max_pool2d_api.py`
  - `test/legacy_test/test_fractional_max_pool3d_api.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，上述四个测试应 fail（op 未注册 / kernel 未实现）。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass。
- P2P 候选：`test/legacy_test/` 下已有的池化相关测试可作为回归护栏，例如 `test_pool2d_api.py`、`test_pool3d_api.py` 等，可由 verifier 自动抽取稳定 nodeid。

## 6. 环境与资源

- 资源需求：CPU + GPU（CUDA kernel 涉及 GPU 编译）
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无，建议后续补充 source-build Dockerfile
- patch 类型：含 C++ CPU kernel + CUDA GPU kernel + Python API + YAML op 定义 + infermeta
- 环境建议：该样本涉及 C++ 和 CUDA kernel，需要 source build
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述目标行为和验收标准，不直接指出具体修改行。
- 环境风险：该样本涉及 C++ 和 CUDA kernel，历史 commit 复现可能需要 source build。
- flaky 风险：池化算子通常有随机性，需要固定 seed 并重复运行，由 verifier 抽取稳定 F2P/P2P nodeid。
- 拆分风险：该 PR 的目标集中在新增 `FractionalMaxPool2d` 和 `FractionalMaxPool3d`，适合作为一个样本。
