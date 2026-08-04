# Task Proposal: PaddlePaddle__Paddle-76873

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-76873`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/76873 ；https://github.com/PaddlePaddle/Paddle/pull/77103 （一并纳入）
- PR 标题：`【Hackathon 9th Sprint No.4: Partial】paddle.nn.* api support inplace operation -part` / `【Hackathon 9th Sprint No.4】paddle.nn.* api support inplace operation`
- `base_commit`：`471930236df5ba4e3bc34e1af6b8b9118e55a2d2`（#76873 squash 合入的父提交）
- gold endpoint：`231207ce894f7f13e5c68e24cfa251ad41d10532`（#77103 合入后）
- merged 时间：`2025-12-22`（#76873）、`2026-01-13`（#77103）
- 你的身份：原 PR 作者（GitHub @yangguohao）
- 后续联系人：GitHub @yangguohao

## 2. 问题一句话

为若干 `paddle.nn` / `paddle.nn.functional` 激活 API 补齐 inplace 能力（含 `CELU`、`RReLU`、`Swish`、`Mish`、`HardSigmoid`、`SELU`），使动态图与相关静态 / 符号形状路径行为一致且可测。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 Hackathon 9th Sprint No.4 框架开发 PR（先 partial 合入 `CELU`，再补齐其余激活），不是合成任务。
- **代表性**：覆盖激活 API 的 inplace 语义、算子配置与 PIR 符号形状一致性，属于常见的 API 兼容 / 体验增强类改造。
- **边界清楚**：目标是上述激活在非 inplace 与 inplace 路径下结果一致、既有非 inplace 行为不被破坏；不要求一次覆盖全部 `paddle.nn.*` 激活。
- **非平凡性**：不仅是 Python 层加一个 `inplace` 开关，还需要算子侧与符号形状路径对齐，否则动态图可用但静态 / CINN 相关路径会缺覆盖或行为不一致。
- **区分度潜力**：只改 Python API、或只改测试期望、或漏掉某一激活 / 符号形状路径，都会被完整验收拦住。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, activation, inplace, api_compatibility, pir, symbolic_shape, legacy_test]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：
  - `test/legacy_test/test_celu_op.py`
  - `test/legacy_test/test_rrelu_op.py`
  - `test/legacy_test/test_swish_op.py`
  - `test/legacy_test/test_mish_op.py`
  - `test/legacy_test/test_hardsigmoid_op.py`
  - `test/legacy_test/test_selu_op.py`
  - 以及 symbolic shape 相关用例（如 `test/ir/pir/cinn/symbolic/test_infer_sym_shape_multinary_op.py` / `test_infer_sym_shape_unary_op.py` 中与上述激活 inplace 相关的部分）
- 修复前预期：在 `base_commit` 上应用 `tests/test.patch` 后，上述激活的 inplace / 兼容性 F2P 用例 fail 或 error；既有非 inplace 行为对应的 P2P 可通过。
- 修复后预期：继续应用 `solution/code.patch` 并按需重新构建后，F2P 与 P2P 均通过。
- P2P 护栏：同模块中原有非 inplace 激活数值与梯度用例继续通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- 是否需要 GPU：否
- patch 类型：含 Python API + 算子配置 / PIR 符号形状相关改动，**需 source build**（不能只依赖已有 wheel overlay）
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只写问题、复现、期望行为与验收标准；不包含衍生 PR 链接、diff、具体修改文件、具体实现步骤或答案路径。
- 环境风险：涉及算子配置与 C++ 符号形状路径，历史 commit 复现需要 source build；symbolic / CINN 相关用例对环境要求更高，打包时可按 verifier 能力裁剪稳定 F2P 子集。
- flaky 风险：`RReLU` 等含随机性的用例需固定 seed；verifier 应抽取稳定 nodeid。
- 拆分风险：两 PR 同属 Hackathon 9th Sprint No.4 的 inplace 激活补齐，合并为一个样本更合理；gold 取相对 `base_commit` 的净效果（#76873 + #77103）。
