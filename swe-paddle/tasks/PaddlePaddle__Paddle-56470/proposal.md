# Task Proposal: PaddlePaddle__Paddle-56470

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-56470`
- Issue 链接：https://github.com/PaddlePaddle/Paddle/issues/55883
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/56470
- PR 标题：``[API Enhancement] No.6 support single `int` input in UpsamplingNearest2D and UpsamplingBilinear2D``
- `base_commit`：`3568a99c5f6ff0e5fd528d43bd283fde34fe078b`
- merged 时间：`2023-08-28`
- 你的身份：原 PR 作者 / reviewer / 熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为 `UpsamplingNearest2D` 和 `UpsamplingBilinear2D` 增加单个整数 `size` 输入支持，并将其解释为正方形输出尺寸。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该任务来自已合入的 Paddle API 易用性增强 PR，不是合成任务。
- **代表性**：它覆盖公共 layer API 的参数规范化和既有输入形式兼容性。
- **边界清楚**：production 行为集中在两个 2D Upsampling layer 对 `size` 的处理。
- **非平凡性**：修复需要同时覆盖 nearest 与 bilinear 两个 API，并保持 list、tuple 和 `scale_factor` 语义不变。
- **环境友好性**：目标逻辑为 Python-only，可通过 checkout source AST overlay 在 CPU 环境稳定验证，无需 source build。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[nn, layer_api, upsampling, interpolation, python_only, api_usability]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr56470_upsampling_single_int.py`
- 修复前预期：list、tuple 和 `scale_factor` 的既有行为通过；两个 layer 的单整数 `size` 测试失败。
- 修复后预期：继续应用 `solution/code.patch` 后，全部目标测试通过。
- P2P 候选：两个 layer 对 list/tuple `size` 及 `scale_factor` 的既有参数传递行为。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：从 checkout source 提取两个 layer 的真实 class control flow，并以 controlled functional double 观察 `interpolate` 参数，无需导入完整历史 Paddle 模块。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 仅描述公共 API 行为，不指出 Gold patch 的具体代码位置。
- 环境风险：AST overlay 避免历史 source 与当前 Paddle wheel 的整体 import 兼容问题。
- flaky 风险：测试仅验证确定性的参数规范化和调用 contract，不依赖随机数、GPU 或网络。
- 拆分风险：两个 layer 共享同一 API enhancement，适合作为一个样本。
