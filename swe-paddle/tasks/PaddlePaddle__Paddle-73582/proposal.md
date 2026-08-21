# SWE-Paddle Task Proposal: PaddlePaddle__Paddle-73582

## 1. 来源信息

- Instance ID: `PaddlePaddle__Paddle-73582`
- PR 链接: https://github.com/PaddlePaddle/Paddle/pull/73582
- PR 标题: `[0-size Tensor Job2 No.12、77] Add 0-size Tensor support for paddle.squeeze/full`
- Base commit: `ecd685afb0ffc1f509771cd1820254c8b42020ad`
- Gold commit: `f69b42e57712ab1c68edc071bee41758c27612f7`
- Merged at: 2025-06-30
- 你的身份: contributor

## 2. 问题一句话

`paddle.squeeze` 在 axis 为 0-size Tensor 时行为异常，`paddle.full` 在 shape 包含 0-size Tensor 时报错，需要在 Python 层添加 0-size 处理逻辑。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**: 来自 Paddle「0-size Tensor 机制建设」系列任务，是真实研发需求。
- **代表性**: 覆盖 Python API 层的 0-size Tensor 边界处理，涉及 squeeze 和 full 两个 API。
- **边界清楚**: 目标仅限 0-size Tensor 输入的特殊处理；正向非零尺寸输入不应受影响。
- **非平凡性**: 修复需要理解 squeeze 的 axis 处理流程和 full 的 shape 转换逻辑，在正确位置添加 0-size 检查。
- **回归护栏明确**: 目标 F2P 可覆盖 0-size Tensor 输入的 squeeze/full 测试；同文件中已有的标准测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型: `bug_fix`
- 执行后端: `cpu`
- 设备范围: `cpu_only`
- 模块标签: `[python_api, squeeze, full, 0-size_tensor, tensor_manipulation]`

## 5. 验证思路

- 目标测试命令: `bash tests/test.sh`
- 目标测试文件:
  - `test/legacy_test/test_fill_constant_op.py`（`TestFillConstantOp_ZeroSize`）
  - `test/legacy_test/test_squeeze2_op.py`（`TestSqueezeAPI_ZeroSize`）
- P2P 候选: 同文件中已有的 `TestFillConstantOp`、`TestSqueezeAPI` 等标准测试用例。
- 修复前预期: `base_commit` + `tests/test.patch` 后，0-size Tensor 输入的 squeeze/full 测试失败。
- 修复后预期: 继续应用 `solution/code.patch` 后，0-size Tensor 输入正常处理，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker: 无
- Dockerfile 或镜像地址: 暂无
- Paddle 来源: `PaddlePaddle/Paddle` source checkout at `base_commit`，需要源码编译（或安装对应版本的 Paddle）。
- OS / Python / CUDA / cuDNN / 其他关键依赖: Linux CPU + Python + numpy；编译需要 CMake、GCC；不要求 CUDA/cuDNN（CPU 编译即可验证）。
- 硬件: CPU 即可（编译和测试均不需要 GPU）。
- patch 类型: 仅 Python 代码修改（`python/paddle/tensor/manipulation.py` 和 `python/paddle/utils/layers_utils.py`），无需重新编译。
- 最小测试命令: `bash tests/test.sh`
- 是否有 oracle 日志: 无

## 7. 风险自查

- 泄露风险: 正式 `instruction.md` 只描述「squeeze/full 对 0-size Tensor 输入的行为异常」，不指出具体的代码位置或实现细节。
- 环境风险: 中。任务需要 Paddle 源码环境，但 patch 仅涉及 Python 文件，无需重新编译。
- flaky 风险: 低。测试使用固定的 tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险: 低。该 PR 目标集中在两个 API 的 0-size Tensor 处理，测试明确指向新增的 ZeroSize 测试类，适合作为一个独立样本。
- 其他不确定点: 完整任务包阶段应确认新增 F2P 在 `base_commit` 上确实失败。
