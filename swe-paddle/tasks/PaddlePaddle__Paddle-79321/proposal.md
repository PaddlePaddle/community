# Task Proposal: PaddlePaddle__Paddle-79321

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79321`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79321
- PR 标题：`[API Compatibility] Return _IncompatibleKeys for set_state_dict`
- `base_commit`：`3cb4059b8e870c818031779af94eae728177c2ac`
- merged 时间：`2026-06-18T08:46:05Z`
- 你的身份：contributor

## 2. 问题一句话

对齐 `paddle.nn.Layer.set_state_dict` 的返回值语义，使其在保持原有二元组解包行为的同时，支持通过 `missing_keys` 和 `unexpected_keys` 命名字段访问。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自 Paddle 主仓已合入的 API Compatibility PR，目标是补齐公开 API 的用户可见返回值语义。
- **代表性**：它覆盖神经网络 `Layer` 状态字典加载、序列化兼容和 PyTorch 风格返回对象，是框架 API 兼容任务中的典型轻量样本。
- **边界清楚**：目标只涉及 `set_state_dict` 的返回对象；参数加载、strict 检查、权重赋值和已有 `load_state_dict` 行为不应被改动。
- **非平凡性**：修复不能简单改变返回类型而破坏旧代码的 tuple 解包兼容性，需要同时满足向后兼容和命名字段访问。
- **验证成本低**：生产修改为 Python 代码，目标测试使用 CPU 即可，不依赖 GPU、分布式、多进程、外部服务或 C++ 重编译。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, layer, state_dict, serialization, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_state_dict_convert.py`
- F2P nodeid：建议新增 `test/legacy_test/test_state_dict_convert.py::TestLoadStateDict::test_missing_keys_and_unexpected_keys_attr`
- P2P 候选：同文件中已有 `test_missing_keys_and_unexpected_keys` 等依赖 tuple 解包的状态字典回归用例。
- 修复前预期：P2P 通过；新增 F2P 用例在 `base_commit` + `tests/test.patch` 上失败，因为 `set_state_dict` 返回普通 tuple，不能通过 `result.missing_keys` / `result.unexpected_keys` 访问。
- 修复后预期：继续应用 `solution/code.patch` 后，返回值既可按二元组解包，也可通过命名字段访问；F2P 与 P2P 均通过。

## 6. 环境与资源

- 是否能提供 Docker：无
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，可使用 base 兼容的本地 Paddle 构建并叠加 Python 源码修改。
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：可由 verifier 选择与 base 兼容的 CPU wheel 或本地源码环境；proposal 阶段不固定 wheel URL。
- OS / Python / CUDA / cuDNN / 其他关键依赖：Linux CPU + Python + pytest；不要求 CUDA/cuDNN。
- 硬件：CPU 即可。
- patch 类型：纯 Python 修改 + Python legacy test，无需 C++ rebuild。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：无；由 SWE-Paddle verifier 记录 Run/Test/Fix 结果。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述返回值的可观察兼容语义，不指出具体 helper 或内部类定义位置。
- 环境风险：低。任务为 Python-only，但 verifier 仍需确保测试加载的是 base checkout 中被 patch 的 Python 源码。
- flaky 风险：低。测试只构造小型 `Layer` 和状态字典，不依赖随机数、外部数据或设备差异。
- 拆分风险：低。该 PR 目标集中在 `set_state_dict` 返回值兼容，适合作为一个独立样本。
- 其他不确定点：完整任务包阶段应确认 `load_state_dict` 相关既有行为不被纳入额外目标，只作为回归上下文。