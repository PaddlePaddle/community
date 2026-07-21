# 任务提案：PaddlePaddle__Paddle-58219

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-58219`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/58219
- PR 标题：`[Hackathon 5th No.49][pir] add some operation - Part 3`
- `base_commit`：`4dbd3f7d8a3a54066939a2e1acc46fadadf65c11`（GitHub PR REST 返回的 `base.sha`）
- PR head：`d682fcc85174d9921f7f4222359ed39faeb9e8ac`
- merged 时间：`2023-10-23T03:41:44Z`（merge commit `edcfda9cd60bb6985bd34851e4501c7743d11d38`）
- 你的身份：原 PR 作者（GitHub @gouzil）
- 后续联系人：GitHub @gouzil

## 2. 问题一句话

PIR 静态图中的 `OpResult` 缺少幂、整除、取模和矩阵乘法运算符，使用 `**`、`//`、`%`、`@` 会直接报 `TypeError`；同时 `floor_divide` 未识别 PIR 模式，会错误进入旧静态图分支。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自 Paddle Hackathon 5th No.49 的已合入 PR，是补齐 PIR `OpResult` Python 运算能力的真实开发任务，并经过 Paddle reviewer 审核。
- **代表性**：覆盖 PIR 静态图、Python 运算符重载、`OpResult` monkey patch、动态图/PIR 模式分发和 executor 数值验证，代表 Paddle 从旧静态图迁移到 PIR 时常见的 API 语义补齐工作。
- **边界清楚**：gold patch 只修改两个 Python 实现文件，目标是为 `OpResult` 注册 `__pow__`、`__rpow__`、`__floordiv__`、`__mod__`、`__matmul__`，并让 `floor_divide` 进入 PIR 分支；不修改 C++、kernel 或算子数值实现，也不包含同系列前两段 PR 已提供的 monkey-patch 基础设施。
- **非平凡性**：模型需要同时找到共享的二元运算方法生成路径和 `floor_divide` 的模式分发问题。只补运算符名称而不修正 PIR 分支，或只改公开函数而不注册 `OpResult` dunder，均不能通过完整测试。
- **验收边界**：沿用原 PR 已覆盖的 `OpResult`-`OpResult` 运算和 `x.pow(2)`；原测试中注释掉的标量反向幂场景不扩展为本任务要求。

## 4. 任务类型和标签

- 任务类型：`feature_implementation`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, pir, static_graph, opresult, operator_overload, monkey_patch, legacy_test]`

## 5. 验证思路

- 目标测试文件：`test/legacy_test/test_math_op_patch_pir.py`
- 目标测试命令：

  ```bash
  python test/legacy_test/test_math_op_patch_pir.py -v
  ```

- F2P 用例：
  - `TestMathOpPatchesPir.test_pow`
  - `TestMathOpPatchesPir.test_floordiv`
  - `TestMathOpPatchesPir.test_mod`
  - `TestMathOpPatchesPir.test_matmul`
- 修复前预期：在 `base_commit` 上应用独立的测试补丁后，四个 F2P 均因 `OpResult` 不支持对应运算符而报 `TypeError`；四个存量 P2P 继续通过。
- 修复后预期：继续应用仅含两个非测试文件改动的 gold patch 后，目标文件中的 8 个用例全部通过，四种运算的 PIR executor 结果与 NumPy/动态图结果一致。
- P2P 候选：原文件中未被 PR 修改的 `test_item`、`test_place`、`test_some_dim`、`test_math_exists`。
- 已完成的兼容性 Run/Test/Fix 预验证：base 为 `4 errors + 4 passes`，应用 gold 后为 `8 passes`；补丁通过 `git diff --check`。

## 6. 环境与资源

- 是否能提供 Docker：有；proposal 预验证使用官方 CPU 镜像
- Dockerfile 或镜像地址：`paddlepaddle/paddle:2.6.0`，以 `--platform linux/amd64` 运行
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`；测试补丁和 gold patch 来自 #58219 的三文件 PR diff
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：未找到与 `base_commit` 精确对应的历史 wheel；镜像内为 Paddle `2.6.0`（commit `e032331bf78b0f9b51806c6761254c8b977f02b4`）、Python `3.10.13`、Linux x86_64
- OS / Python / CUDA / cuDNN / 其他关键依赖：Linux x86_64、Python 3.10、NumPy、`unittest`；CPU 测试，不依赖 CUDA/cuDNN
- 硬件：CPU；本次在 macOS arm64 主机上通过 Docker amd64 仿真验证
- patch 类型：纯 Python，不含 C++、CUDA、kernel 或 infermeta 编译改动
- 最小测试命令：`python test/legacy_test/test_math_op_patch_pir.py -v`
- 是否有 oracle 日志：有本次本地预验证输出；完整任务包阶段仍需由 SWE-Paddle verifier 归档正式日志
- 兼容性说明：2.6.0 镜像的 Python 层已包含后续 PIR 命名变化，本次预验证仅在运行时适配 `monkey_patch_opresult` / `monkey_patch_value` 名称，并分别加载 PR 前后的目标逻辑；适配不进入 gold patch。完整任务包阶段应优先 source build 精确基线，或把该兼容环境固化后再复跑。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述四种运算的可观察行为、PIR 静态图执行和数值结果，不应给出 monkey-patch 列表、共享 helper 或具体修改文件。
- 环境风险：精确历史 wheel 已不在当前 nightly 索引。虽然 gold patch 是纯 Python，PIR Python API 与编译产物之间存在版本耦合，完整 verifier 仍需精确 source build 或固定兼容镜像。
- flaky 风险：低。测试显式固定 CPU，整数除数非零，数值比较使用 `assert_allclose`；随机输入未固定 seed，但不依赖概率阈值或训练收敛。
- 拆分风险：PR 同时加入四种运算，但它们共享同一 `OpResult` 二元运算注册机制和同一测试入口，按 PR 粒度保留为一个样本更自然。若拆分，会人为切割同一 gold patch。
- 依赖风险：任务依赖 #57857、#58118 等同系列前序工作已存在于 base；这些基础能力不应重复纳入本任务。
- patch 提取风险：GitHub 记录的 `base.sha` 与 PR head 不在同一线性祖先链上，直接执行 `git diff base_commit..head` 会混入 31 个无关文件。gold/test patch 必须从 GitHub 三文件 PR diff（等价于 merge-base `5e3974e5b391999f28e326b8c97f23284741934d` 到 head 的目标文件 diff）提取，并确认可干净应用到 `base_commit`。
