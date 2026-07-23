# Task Proposal: PaddlePaddle__Paddle-79161

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79161`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79161
- PR 标题：`[API Compatibility] Add param alias for paddle.set_rng_state`
- `base_commit`：`8dd02b271734f7aae3669fe6dbcbea57d9cc9add`
- merged 时间：`2026-05-28`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为 `paddle.set_rng_state` 的 `state_list` 参数增加等价别名 `new_state`，提升与其他框架风格调用的兼容性。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle API Compatibility PR，不是合成任务。
- **代表性**：它覆盖公共 Python API 的参数 alias、向后兼容和冲突参数语义。
- **边界清楚**：production 行为集中在 `paddle.set_rng_state` 的调用签名兼容性。
- **非平凡性**：实现不仅要接受新 alias，还必须保留 positional 与原关键字调用，并正确拒绝同时传入 canonical 参数和 alias 的冲突场景。
- **环境友好性**：目标逻辑为 Python-only，可通过 checkout source AST overlay 在 CPU 环境稳定验证，无需 source build。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[random, api_compatibility, parameter_alias, decorator, python_only]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr79161_set_rng_state_alias.py`
- 修复前预期：positional 和 `state_list=` 既有调用通过；`new_state=` alias 与 alias 冲突语义测试失败。
- 修复后预期：继续应用 `solution/code.patch` 后，全部目标测试通过。
- P2P 候选：positional state list、原 `state_list=` 关键字和显式 CPU device 参数的既有行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：从 checkout source 提取 `set_rng_state` 的真实 function 与 decorator control flow，并以 controlled CPU place 和 generator doubles 观察 state side effect，无需导入完整历史 Paddle 模块。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 仅描述公共 API 行为，不指出具体 decorator 或修改行。
- 环境风险：AST overlay 避免历史 source 与当前 Paddle wheel 的整体 import 兼容问题。
- flaky 风险：测试只验证确定性的参数映射、异常类型和 CPU generator side effect，不依赖随机采样结果。
- 拆分风险：alias 支持及其冲突语义属于同一个公共 API compatibility 目标，适合作为一个样本。
