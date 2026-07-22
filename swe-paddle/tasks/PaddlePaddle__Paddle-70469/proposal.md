# Task Proposal: PaddlePaddle__Paddle-70469

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-70469`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/70469
- PR 标题：`[BugFix] Fall back fused dropout add`
- `base_commit`：`052874d0c95fe9bcae0c3e0ac60d857b238e6d70`
- merged 时间：`2024-12-27`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 `fused_dropout_add` 在已知 precision issue 尚未解决时仍进入 fused execution path，导致结果可能偏离标准 `dropout + add` semantics 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle BugFix PR，不是合成任务。
- **代表性**：它覆盖 fused functional API、operator dispatch、fallback semantics、参数传递和 warning lifecycle。
- **边界清楚**：目标行为集中在 `fused_dropout_add` 的 Python API fallback，不需要修改底层 CUDA kernel 或 distributed runtime。
- **非平凡性**：这个任务不只是替换一个 operator call。正确修复还需要保留 `p`、`training` 和 `mode` semantics，并实现 warning-once behavior。
- **确定性**：verifier 可以使用 controlled dropout / fused-op doubles 验证 output、参数传递、dispatch side effect 和 warning count，不依赖真实 dropout randomness、GPU kernel 或统计容差。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[fused_dropout_add, dropout, fallback, precision, warning, python_api]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_fused_dropout_add_fallback.py`
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，training fallback、inference fallback 和 warning-once 相关测试应 fail。
- 修复后预期：继续应用 `solution/code.patch` 后，目标测试应 pass；输出应符合 controlled `dropout + add` reference，参数应被正确传递，受影响的 fused execution path 不应被调用，且 warning 在同一 loaded module 中最多发出一次。
- P2P 候选：`p == 0` 等已有有效调用，可由 verifier 作为 regression guard，确认 fallback 不会破坏既有 semantics。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production change
- 环境建议：通过 AST 加载 source checkout 中的目标 function，并使用 controlled dropout / fused-op doubles，避免依赖已安装 Paddle wheel 或真实 fused kernel
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 应描述 observable behavior、需要暂时避开的 affected execution path 和 warning contract，不直接指出目标文件、helper function、global state 名称或具体修改行。
- spec-test alignment 风险：如果 verifier 检查 fused dispatch side effect，正式 instruction 必须明确说明在 precision issue 解决前应避开受影响的 fused execution path。
- flaky 风险：测试应使用 controlled doubles 返回固定结果，不依赖真实 dropout randomness 或浮点统计容差。
- 环境风险：该样本为 Python-only production change，可通过 source-level function loading 完成验证，不依赖 GPU、CUDA fused kernel 或 distributed launcher。
- 误判风险：verifier 应以 behavioral tests 和 regression constraints 为准，不要求源码与 `gold_commit` 完全一致，也不检查特定 helper、global flag、AST 结构或精确修改行数。
- 投机风险：training、inference、不同参数组合和 `p == 0` regression case 应分别覆盖，避免 agent 通过 hard-code 单一路径通过测试。