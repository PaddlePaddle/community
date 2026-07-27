# Task Proposal: PaddlePaddle__Paddle-79275

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79275`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79275
- PR 标题：`[API Compatibility] Align torch.nn.attention.flex_attention.or_masks/and_masks`
- `base_commit`：`1d14ac949cd00747df9c828537f5fbff51b1f85f`
- merged 时间：`2026-06-12`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为 flex attention 增加可组合多个 mask callable 的 `or_masks` / `and_masks` 公共 API，并对齐单 mask、空输入和非法输入的可观察行为。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle PR #79275，不是合成需求。
- **代表性**：这是 attention 模块的公共 API addition，可补充以 bug fix 为主的任务类型。
- **边界清楚**：Gold production change 只涉及 attention package 导出和新增的 `flex_attention.py`，目标行为可由独立测试直接观察。
- **非平凡性**：正确实现需要同时满足多 callable OR/AND 组合、参数透传、单 mask 与空输入恒等语义以及非法输入校验。
- **环境友好性**：组合控制流可以通过 AST overlay 与 Tensor-like controlled doubles 在 CPU 上稳定验证，无需真实 attention kernel、GPU 或 Paddle source build。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[attention, flex_attention, public_api, api_compatibility, mask, python_only]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr79275_flex_attention_masks.py`
- 修复前预期：现有 attention export P2P 通过；`or_masks` / `and_masks` 的组合、恒等和错误输入 F2P 因目标 API 尚不存在而失败。
- 修复后预期：继续应用 production-only `solution/code.patch` 后，全部目标测试通过。
- P2P 候选：`paddle.nn.attention` 原有 `SDPBackend`、`sdpa_kernel` 与公开导出集合保持不变。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：通过 AST overlay 执行 checkout 中真实 `or_masks` / `and_masks` 控制流，并使用 controlled Tensor-like doubles 观察逻辑组合、参数透传和 identity 初始化行为；无需 source build。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述公共 API 的 observable behavior，不给出 Gold patch 的具体实现步骤。
- 环境风险：测试不导入历史 checkout 的完整 Paddle package，不要求 native extension 与源码版本完全匹配。
- flaky 风险：所有 mask 输入和布尔结果由 controlled doubles 固定，不依赖随机数、GPU 或并发时序。
- 拆分风险：`or_masks` 和 `and_masks` 是同一 flex-attention mask composition API 目标，且由同一新增模块提供，适合作为一个任务。
