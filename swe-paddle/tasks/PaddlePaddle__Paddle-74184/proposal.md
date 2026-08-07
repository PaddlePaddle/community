# SWE-Paddle Task Proposal: PaddlePaddle__Paddle-74184

## 1. 来源信息

- Instance ID: `PaddlePaddle__Paddle-74184`
- PR 链接: https://github.com/PaddlePaddle/Paddle/pull/74184
- PR 标题: `[0-size Tensor No.114] Add 0-size Tensor support for paddle.linalg.pinv`
- Base commit: `ac82c42a5c17f1ddd3ac50a28bb8d0ce84acba8e`
- Gold commit: `ea80fa17d84889795799fa5f868572b24bd8837c`
- Merged at: 2025-07-24
- 你的身份: contributor

## 2. 问题一句话

`paddle.linalg.pinv` 在输入为 0-size Tensor 时，SVD 路径对空奇异值取 max 报错，hermitian 路径缺少 0-size 快速返回逻辑，需要分别添加 0-size 边界处理。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**: 来自 Paddle「0-size Tensor 机制建设」系列任务，是真实研发需求。
- **代表性**: 覆盖 Python API 层面的 0-size Tensor 边界处理，涉及 SVD 和 eigh 两条代码路径的修复。
- **边界清楚**: 目标仅限 0-size Tensor 输入时 pinv 函数的两条路径（hermitian=False/True）；正向非零尺寸输入不应受影响。
- **非平凡性**: 修复需要在 SVD 路径中判断 `s.shape[-1] == 0` 时对 `max_singular_val` 做特殊赋值，在 hermitian 路径中判断 `x.size == 0` 时直接返回转置结果，涉及对伪逆数学语义和 0-size Tensor 行为的理解。
- **回归护栏明确**: 目标 F2P 可覆盖 0-size Tensor 输入的 `pinv` 测试（`LinalgPinvTestCase_ZeroSize` 和 `LinalgPinvTestCaseHermitian_ZeroSize`）；同文件中已有的 `LinalgPinvTestCase` 等标准测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型: `bug_fix`
- 执行后端: `cpu`
- 设备范围: `cpu_only`
- 模块标签: `[python_api, linalg, pinv, 0-size_tensor, svd, eigh]`

## 5. 验证思路

- 目标测试命令: `bash tests/test.sh`
- 目标测试文件:
  - `test/legacy_test/test_linalg_pinv_op.py`（`LinalgPinvTestCase_ZeroSize`、`LinalgPinvTestCaseHermitian_ZeroSize`）
- P2P 候选: 同文件中已有的 `LinalgPinvTestCase`、`LinalgPinvTestCaseHermitian` 等标准 pinv 测试用例。
- 修复前预期: `base_commit` + `tests/test.patch` 后，0-size Tensor 输入的 `pinv` 测试失败（SVD 路径或 eigh 路径报错）。
- 修复后预期: 继续应用 `solution/code.patch` 后，0-size Tensor 输入正常返回正确结果，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker: 无
- Dockerfile 或镜像地址: 暂无
- Paddle 来源: `PaddlePaddle/Paddle` source checkout at `base_commit`。
- OS / Python / CUDA / cuDNN / 其他关键依赖: Linux CPU + Python + numpy；本任务仅修改 Python 代码，无需重新编译 C++。
- 硬件: CPU 即可（测试不需要 GPU）。
- patch 类型: 仅 Python 修改（`python/paddle/tensor/linalg.py`），无需重新编译 Paddle。
- 最小测试命令: `bash tests/test.sh`
- 是否有 oracle 日志: 无

## 7. 风险自查

- 泄露风险: 正式 `instruction.md` 只描述「pinv 对 0-size Tensor 输入的行为异常」，不指出具体 `s.shape[-1] == 0` 或 `in_dynamic_mode() and x.size == 0` 分支逻辑。
- 环境风险: 低。任务仅涉及 Python 修改，不需要源码编译 Paddle。
- flaky 风险: 低。测试使用固定种子的 0-size Tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险: 低。该 PR 目标集中在 `pinv` 函数的 0-size 边界处理，测试明确指向 `LinalgPinvTestCase_ZeroSize` 和 `LinalgPinvTestCaseHermitian_ZeroSize`，适合作为一个独立样本。
- 其他不确定点: 完整任务包阶段应确认新增 F2P 在 `base_commit` 确实失败；同时注意 `TestDivByZero.test_div_by_zero` 的修改（注释掉原有异常断言）。
