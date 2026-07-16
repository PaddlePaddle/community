# Task Proposal: PaddlePaddle__Paddle-79386

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79386`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79386
- PR 标题：`[API Compatibility] Bug Fix`
- `base_commit`：`9aa3379edbee8ccd6cec772b22ad37733357f8df`
- merged 时间：`2026-06-29T08:47:50Z`
- 你的身份：contributor

## 2. 问题一句话

修复 `paddle.iinfo(paddle.uint64).max` 在 Python 侧被错误解释为有符号整数的问题，确保 `uint64` 最大值返回 `18446744073709551615`，而不是 `-1`。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自 Paddle 主仓已合入的 API Compatibility bug-fix PR，源于 PaConvert / API 边界测试中暴露的真实 dtype 边界问题。
- **代表性**：它覆盖 Paddle C++ pybind 到 Python API 的整数边界值暴露，尤其是无符号 64 位整数跨语言绑定时的符号解释问题。
- **边界清楚**：目标行为集中在 `paddle.iinfo` 的 `uint64.max`；其他整数 dtype 的 `min`、`max`、`bits` 和 `dtype` 行为必须保持不变。
- **非平凡性**：修复需要理解 C++ `uint64_t` 与 Python integer 绑定的类型转换边界，不能通过修改 Python 包装层或硬编码单测返回值来规避。
- **回归护栏明确**：新增 `uint64` 最大值 F2P 后，可用已有 `iinfo` / integer dtype 兼容测试作为 P2P，确保其他 dtype 不被破坏。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, pybind, dtype, iinfo, uint64, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：建议在 `test/legacy_test/test_iinfo_and_finfo.py` 或 `test/legacy_test/test_api_compatibility_part1.py` 中补充 `uint64` 最大值断言。
- F2P nodeid：建议新增一个窄范围用例，验证 `paddle.iinfo(paddle.uint64).max == 18446744073709551615`，同时覆盖字符串或别名形式（如 `'uint64'`）时的行为。
- P2P 候选：`test/legacy_test/test_api_compatibility_part1.py::TestIinfoAPI` 以及已有 integer dtype 属性测试，可验证 `int32`、`uint8`、`uint16`、`uint32` 等既有行为不变。
- 修复前预期：P2P 通过；新增 F2P 在 `base_commit` + `tests/test.patch` 上失败，因为 Python 侧 `uint64.max` 被解释为 `-1` 或不等于无符号 64 位最大值。
- 修复后预期：继续应用 `solution/code.patch` 并使用包含该 pybind 修改的 Paddle 构建后，新增 F2P 通过，其他整数 dtype 的 P2P 回归保持通过。

## 6. 环境与资源

- 是否能提供 Docker：无
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不使用历史 wheel；该任务修改 C++ pybind，需要 source build 或等价增量编译环境。
- OS / Python / CUDA / cuDNN / 其他关键依赖：Linux CPU + Python + pytest；不要求 CUDA/cuDNN。
- 硬件：CPU 即可。
- patch 类型：含 C++ pybind 修改 + Python legacy test。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：无；由 SWE-Paddle verifier 记录 Run/Test/Fix 结果。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述 `paddle.iinfo` 对 `uint64` 最大值的可观察行为，不指出具体 pybind 绑定行或类型转换实现方式。
- 环境风险：需要能够加载重新构建后的 `libpaddle`；不能只用未重建的旧 wheel 验证 gold patch。
- flaky 风险：低。测试为确定性 dtype 属性断言，不依赖随机数、外部数据、GPU 或网络。
- 拆分风险：低。该 PR 目标集中在 `uint64` 的 `iinfo.max` 绑定行为，适合作为一个独立样本。
- 其他不确定点：完整任务包阶段应确认 base commit 上新增 F2P 的实际失败表现，并避免把 PaConvert 相关上下文扩展为额外验收目标。