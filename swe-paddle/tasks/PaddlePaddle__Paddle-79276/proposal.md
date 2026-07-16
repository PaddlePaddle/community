# Task Proposal: PaddlePaddle__Paddle-79276

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79276`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79276
- PR 标题：`fix add_n 0-size bug`
- `base_commit`：`8cacdfd15bc89296682c784df5b1685a7ca6e408`
- merged 时间：`2026-06-09T08:17:29Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 `paddle.add_n` 在输入列表中同时包含 0-size Tensor 和非 0-size Tensor 时，可能因第一个 0-size Tensor 被误当作未初始化 shape 而跳过后续 shape 全等校验的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自 Paddle 主仓已合入的真实 operator mechanism bug-fix PR，不是合成问题。
- **代表性**：它覆盖 Paddle infermeta 中 0-size Tensor 边界、运行期 shape 校验和 `add_n` 多输入语义，是深度学习框架常见的边界形状问题。
- **边界清楚**：目标行为限定为 `add_n` 多输入 shape 校验；合法的全 0-size 输入和普通 shape 一致输入仍应保持原有行为。
- **非平凡性**：正确修复需要区分“尚未记录参考 shape”和“参考 shape 本身 product 为 0”这两种状态，不能简单放宽或删除 shape 校验。
- **回归护栏明确**：目标 F2P 可覆盖 0-size 与非 0-size 混合输入；同文件已有 0-size 合法行为和普通 `add_n` 测试可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[operator_mechanism, infermeta, add_n, zero_size_tensor, shape_validation, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_add_n_op.py`
- F2P nodeid：建议新增 `test/legacy_test/test_add_n_op.py::TestAddnOpZeroSizeAndNonZeroSize::test_add_n_zero_size_and_non_zero_size`
- P2P 候选：同文件中已有的 0-size `add_n` 合法用例（如 `test_add_n_zerosize`）以及普通 `add_n` 前向/反向回归测试。
- 修复前预期：P2P 通过；新增 F2P 用例在 `base_commit` + `tests/test.patch` 上失败，因为 0-size 与非 0-size 输入组合没有按预期抛出 shape mismatch 错误。
- 修复后预期：继续应用 `solution/code.patch` 并使用包含该 infermeta 修改的 Paddle 构建后，新增 F2P 用例通过，P2P 回归测试保持通过。

## 6. 环境与资源

- 是否能提供 Docker：无
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不使用历史 wheel；该任务修改 C++ infermeta，需要 source build 或等价增量编译环境。
- OS / Python / CUDA / cuDNN / 其他关键依赖：Linux CPU + Python + pytest；不要求 CUDA/cuDNN。
- 硬件：CPU 即可。
- patch 类型：含 C++ infermeta 修改 + Python legacy test。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：无；由 SWE-Paddle verifier 记录 Run/Test/Fix 结果。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述 `add_n` 的可观察 shape 校验行为，不直接指出内部状态变量或具体修改行。
- 环境风险：需要能够加载修改后的 C++ infermeta；不能只用未重建的旧 wheel 验证 gold patch。
- flaky 风险：低。测试仅构造小规模 Tensor，不依赖随机数、网络、GPU 或外部数据。
- 拆分风险：低。该 PR 目标集中在 `add_n` 0-size 输入 shape 校验，适合作为一个独立样本。
- 其他不确定点：完整任务包阶段应确认新增 F2P 在 base 上确实失败，并选择一个在 base 与修复后都稳定通过的 P2P nodeid。