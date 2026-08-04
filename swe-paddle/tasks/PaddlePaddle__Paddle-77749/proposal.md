# Task Proposal: PaddlePaddle__Paddle-77749

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-77749`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/77749
- PR 标题：`[API Compatibility] implement nn.utils.rnn.pad_sequence and unpad_sequence`
- `base_commit`：`ea0f979936ab101a91a8739bdb0a528b8df42ef7`（squash 合入 commit `7c19c94684c0e93b6d5f2b288d34d2a61e39b02a` 的第一父提交）
- merged 时间：2026-02-11 16:17:00 UTC
- 你的身份：原 PR 作者（GitHub @Manfredss）
- 后续联系人：GitHub @Manfredss

## 2. 问题一句话

Paddle 缺少与主流深度学习框架兼容的 `paddle.nn.utils.rnn.pad_sequence` 和 `paddle.nn.utils.rnn.unpad_sequence`，变长序列批处理和恢复流程需要用户自行实现。该 PR 新增这两个公开 API，并将其导出到 `paddle.nn.utils`，同时补充完整的单元测试。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自 Paddle 真实的 API Compatibility 新功能 PR，PR #77749 已合入主仓库，作者同时提供了公开 API 实现和对应测试。
- **代表性**：样本覆盖 Python API 设计、变长 Tensor 序列的 shape 处理、batch 布局转换、填充值与填充方向、Tensor 切片恢复以及公开模块导出，代表 Paddle Python API 与常见深度学习工具接口对齐的能力。
- **边界清楚**：`pad_sequence` 接收非空的 Tensor 序列列表，将不同长度的 `L x *` 序列补齐到最长长度；默认输出 `T x B x *`，`batch_first=True` 时输出 `B x T x *`。它需要支持 `padding_value`、`padding_side='right'`/`'left'`，并保留序列的 trailing dimensions 和 dtype。非列表输入应触发 `TypeError`，不支持的填充方向应触发 `ValueError`。`unpad_sequence` 接收 padded Tensor、每个序列的长度 Tensor 和 `batch_first`，返回按原长度切分的 Tensor 列表。两者应正确处理标量尾维、普通多维尾维、整数 dtype、单序列、等长序列，以及 pad/unpad round-trip；不要求引入与本 PR 无关的 RNN 层或训练逻辑。
- **非平凡性**：任务不只是添加两个名称。实现需要同时处理 `T x B x *` 与 `B x T x *` 两种布局、最长序列计算、左/右两侧补齐、多维 trailing shape、填充值 dtype 以及按长度恢复列表等组合行为，并正确接入 `paddle.nn.utils` 的公开导出。错误地交换 batch/time 维度、只支持一维序列或在 round-trip 中保留 padding 都会造成可观察的行为回归。

## 4. 任务类型和标签

- 任务类型：`feature_implementation`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, api_compatibility, rnn, sequence, padding, tensor_shape, dtype, dynamic_graph]`

## 5. 验证思路

- 目标测试文件：
  - `test/legacy_test/test_rnn_utils.py`

- 目标测试命令：

  ```bash
  python -m pytest test/legacy_test/test_rnn_utils.py -q
  ```

- 修复前预期：在 `base_commit + test_patch` 下，测试文件可以加载，但从 `paddle.nn.utils.rnn` 导入新增 API 会失败，或新增函数尚未实现而无法满足行为测试。已有 Paddle 测试应保持可运行，不应通过删除、跳过或弱化新增断言来制造通过结果。
- 修复后预期：在 `base_commit + test_patch + code_patch` 下，`pad_sequence` 和 `unpad_sequence` 可从 `paddle.nn.utils.rnn` 使用，并满足默认及 `batch_first=True` 的输出布局、左右填充、指定填充值、标量/多维尾维、整数 dtype、输入校验和 round-trip 测试；完整目标测试通过。
- 回归护栏：测试应覆盖默认 `T x B x *` 与 `B x T x *` 输出、`padding_value`、`padding_side='left'`、单序列、等长序列、不同 trailing dimensions、0-d trailing dimensions、整数 Tensor、无效输入类型和无效 padding side；`unpad_sequence` 应覆盖两种 batch 布局、不同长度、多维 Tensor、等长序列和单序列；集成测试应确认 pad 后再 unpad 能恢复原始序列值与 shape。

## 6. 环境与资源

- 是否能提供 Docker：暂无固定 Docker 镜像
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，使用源码工作树应用 test patch 和 solution patch；本任务修改 Python API 文件，需使用与该历史版本匹配的 Paddle 源码或构建产物进行验证
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不依赖外部 wheel；优先使用从 `base_commit` 构建并可导入的 Paddle 包
- OS / Python / CUDA / cuDNN / 其他关键依赖：Linux x86_64、该版本支持的 Python 3、NumPy 和 pytest；目标测试使用 CPU 即可，不需要 CUDA、cuDNN 或外部服务
- 已验证构建环境：原 PR 在 Windows 主机上修改、构建并测试，环境为 Win11 Home、Python 3.12、CMake 3.18.6、VS 2022、CUDA 12.9、cuDNN 9.12.0
- 硬件：目标 verifier 使用 CPU 即可；原 PR 验证机器为 9800X3D + RTX 5070Ti，GPU 不是本任务要求
- patch 类型：Python API、公开模块导出和 Python 单元测试，不涉及 C++ kernel 或设备专属代码
- 最小测试命令：`python -m pytest test/legacy_test/test_rnn_utils.py -q`
- 是否有 oracle 日志：暂无固定 oracle 日志；以目标测试在 base + test patch 下失败、应用 code patch 后通过作为 Run/Test/Fix 验证依据

## 7. 风险自查

- 泄露风险：后续 `instruction.md` 只描述公开 API 的输入、输出、布局和可观察行为，不直接暴露具体实现文件、内部调用顺序或完整 diff；本 proposal 仅提供任务边界和验证要求
- 环境风险：该任务依赖 Paddle 历史版本的 Python 包可导入性；不同 Python 版本或未构建的源码工作树可能导致导入问题，但测试本身不依赖 GPU 或外部网络服务
- flaky 风险：测试使用固定的小型 Tensor 和确定性 shape/value 断言，round-trip 使用有限随机输入时只验证恢复结果；不依赖多卡、网络、时间或随机训练过程，预期无明显 flaky 风险
- 拆分风险：两个函数共同构成变长序列的补齐与恢复 API，测试还验证二者的组合行为；将其拆成两个独立样本会丢失 pad/unpad round-trip 契约，因此保留为一个任务
- 其他不确定点：该 PR 的实现是 Python API 层逻辑，静态图专属行为不是本任务目标；验收应聚焦公开 API 的动态图行为、布局和数值结果，并确认新增导出没有破坏既有 `paddle.nn.utils` API
