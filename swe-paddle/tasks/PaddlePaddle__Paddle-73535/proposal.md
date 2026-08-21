# SWE-Paddle Task Proposal: PaddlePaddle__Paddle-73535

## 1. 来源信息

- Instance ID: `PaddlePaddle__Paddle-73535`
- PR 链接: https://github.com/PaddlePaddle/Paddle/pull/73535
- PR 标题: `[Accuracy diff No.112] Fix accuracy diff for paddle.nn.functional.conv1d API`
- Base commit: `9c1900ce422e3398bfccf95d3d33ba2cfa91faed`
- Gold commit: `f8e6a83f6cb3725902082e7fbb011d7c1e8f6406`
- Merged at: 2025-06-24
- 你的身份: contributor

## 2. 问题一句话

`paddle.nn.functional.conv1d` 在 CPU 上不支持 float16 的 weight/bias，需要在 Python 层将 float16 临时转为 float32 计算后再转回 float16。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**: 来自 PaddleAPITest 精度对比测试发现的真实问题（PFCCLab/PaddleAPITest#306）。
- **代表性**: 覆盖 Python API 层的设备/dtype 兼容性修复，涉及设备检测、dtype 转换和结果还原。
- **边界清楚**: 目标仅限 CPU 设备上 float16 weight/bias 的 conv1d 调用；其他设备和 dtype 不受影响。
- **非平凡性**: 修复需要理解 conv1d 到 conv2d 的转换流程，在正确的位置插入 dtype 转换逻辑，并确保输出 dtype 正确还原。
- **回归护栏明确**: 目标 F2P 可覆盖 CPU float16 conv1d 测试；同文件中已有的 `TestFunctionalConv1DError` 等标准测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型: `bug_fix`
- 执行后端: `cpu`
- 设备范围: `cpu_only`
- 模块标签: `[python_api, conv1d, float16, cpu_compatibility, dtype_conversion]`

## 5. 验证思路

- 目标测试命令: `bash tests/test.sh`
- 目标测试文件:
  - `test/legacy_test/test_functional_conv1d.py`（`TestFunctionalConv1D_CPU_FP16`）
- P2P 候选: 同文件中已有的 `TestFunctionalConv1DError`（含 `Case1`、`Case2`）等标准 conv1d 测试用例。
- 修复前预期: `base_commit` + `tests/test.patch` 后，CPU float16 conv1d 测试失败（CPU conv2d 不支持 float16）。
- 修复后预期: 继续应用 `solution/code.patch` 后，CPU float16 conv1d 正常返回正确结果，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker: 无
- Dockerfile 或镜像地址: 暂无
- Paddle 来源: `PaddlePaddle/Paddle` source checkout at `base_commit`，需要源码编译（或安装对应版本的 Paddle）。
- OS / Python / CUDA / cuDNN / 其他关键依赖: Linux CPU + Python + numpy；编译需要 CMake、GCC；不要求 CUDA/cuDNN（CPU 编译即可验证）。
- 硬件: CPU 即可（编译和测试均不需要 GPU）。
- patch 类型: 仅 Python 代码修改（`python/paddle/nn/functional/conv.py`），无需重新编译。
- 最小测试命令: `bash tests/test.sh`
- 是否有 oracle 日志: 无

## 7. 风险自查

- 泄露风险: 正式 `instruction.md` 只描述「conv1d 在 CPU 上对 float16 weight/bias 的行为异常」，不指出具体的转换代码位置或实现细节。
- 环境风险: 中。任务需要 Paddle 源码环境，但 patch 仅涉及 Python 文件，无需重新编译。
- flaky 风险: 低。测试使用固定的小 tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险: 低。该 PR 目标集中在 `conv1d` 函数的 CPU float16 兼容处理，测试明确指向 `TestFunctionalConv1D_CPU_FP16`，适合作为一个独立样本。
- 其他不确定点: 完整任务包阶段应确认新增 F2P（`TestFunctionalConv1D_CPU_FP16`）在 `base_commit` 上确实失败。
