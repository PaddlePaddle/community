# Task Proposal: PaddlePaddle__Paddle-74444

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-74444`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74444
- PR 标题：`[API compatibility] add paddle nn.functional.dropout1d api`
- `base_commit`：`607dd38aead3118af96495d50b9829c78b2ecfab`
- merged 时间：`2025-08-08`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

Paddle 缺少 `paddle.nn.functional.dropout1d`，因此 2D/3D 一维通道特征无法通过兼容 API 执行按通道 dropout。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源于已合入 Paddle `develop` 的 API compatibility PR。
- **代表性**：覆盖常见的 `nn.functional` API 扩展以及输入 shape/参数校验契约。
- **边界清楚**：目标集中在 2D/3D 输入、channel axis、概率范围、`inplace` 兼容处理和 public export。
- **非平凡性**：2D 输入需要保持用户可见 shape，同时与 3D 输入共享按通道 dropout 语义，并保留既有 dropout helper 行为。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[nn, functional, dropout, api_compatibility]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr74444_dropout1d.py`
- 修复前预期：已有 `dropout2d` P2P 通过；`dropout1d` 不存在，因此 channel-dropout 行为和输入校验 F2P 失败。
- 修复后预期：`dropout1d` 在 public namespace 可见，2D/3D 输入均按 channel axis 调用 dropout，错误输入被拒绝，全部测试通过。
- P2P 候选：执行 checkout 中已有 `dropout2d` 函数体，确认 NCHW/NHWC 的 channel-axis forwarding 未退化。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用 Python + pytest；通过 AST overlay 和 controlled Tensor/dropout doubles 验证 checkout 中真实控制流。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：测试验证公开 API 的 shape、参数、异常和 forwarding contract，不要求具体局部变量或实现步骤。
- 环境风险：低；不导入历史 Paddle native runtime，不需要编译或 GPU。
- flaky 风险：低；随机 dropout 被 controlled double 替代，测试仅检查确定性参数传递和 shape 行为。
- 拆分风险：低；两个 production 文件共同提供同一个 functional API 及其公开导出，适合作为单一任务。
