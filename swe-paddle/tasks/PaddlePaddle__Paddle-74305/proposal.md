# Task Proposal: PaddlePaddle__Paddle-74305

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-74305`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/74305
- PR 标题：`[0-size Tensor No.354、355] Add 0-size Tensor support for unique`
- `base_commit`：`2e4a7184e806f1780f7695be46952d651993ed4e`
- merged 时间：`2025-07-31`
- 你的身份：contributor

## 2. 问题一句话

`paddle.unique` 和 `paddle.unique_consecutive` 在动态图模式下对 0-size tensor（任意维度含有 0）的输入缺少显式处理，导致进入底层算子时出错，需要补齐 0-size tensor 支持。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：该问题来自 Paddle 的「0-size Tensor 机制建设」系列任务，是真实研发需求，目标是为 `unique` 和 `unique_consecutive` 算子补齐 0-size tensor 支持。
- **代表性**：覆盖 Python API 层面的算子边界处理，涉及 dynamic mode 下的 tensor 维度判断和多返回值语义（inverse、counts、index），是 Paddle API 算子机制增强的典型样本。
- **边界清楚**：目标仅限 0-size tensor 输入的动态图早期返回逻辑，返回值为空 tensor 且保持 dtype 一致；正向非零尺寸输入不应受影响。
- **非平凡性**：修复需要在 `manipulation.py` 中同时为两个 API 分别添加 `math.prod(x.shape) == 0` 的早期返回分支，涉及 axis 为 `[]` 的展平情况、返回 dtype（int32/int64）判断、以及多返回值解包，不是简单机械修改。
- **回归护栏明确**：目标 F2P 可覆盖 0-size tensor 输入的 `unique` 和 `unique_consecutive` 动态图调用；同文件已有的标准唯一化算子测试用例可作为 P2P 护栏。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[python_api, manipulation, 0-size_tensor, unique, unique_consecutive, dynamic_mode]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：
  - `test/legacy_test/test_unique.py`（`TestUniqueAPI_ZeroSize`）
  - `test/legacy_test/test_unique_consecutive_op.py`（`TestUniqueConsecutive_ZeroSize`）
- P2P 候选：同文件中已有的 `TestUniqueAPI`、`TestUniqueConsecutive` 等标准唯一化算子测试用例。
- 修复前预期：`base_commit` + `tests/test.patch` 后，0-size tensor 输入在 `paddle.unique` / `paddle.unique_consecutive` 的动态图调用中失败（进入底层算子时 shape 推断异常）。
- 修复后预期：继续应用 `solution/code.patch` 后，0-size tensor 输入返回正确的空 tensor（shape 与预期一致），同时支持 `return_inverse`、`return_counts`、`return_index` 等可选输出，P2P 存量测试仍然通过。

## 6. 环境与资源

- 是否能提供 Docker：无
- Dockerfile 或镜像地址：暂无
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`，纯 Python 修改可直接 patch。
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：可由 verifier 选择与 base 兼容的 CPU wheel 或本地源码环境；proposal 阶段不固定 wheel URL。
- OS / Python / CUDA / cuDNN / 其他关键依赖：Linux CPU + Python + numpy + pytest；不要求 CUDA/cuDNN。
- 硬件：CPU 即可。
- patch 类型：纯 Python 修改 + Python legacy test，无需 C++ rebuild。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：无；由 SWE-Paddle verifier 记录 Run/Test/Fix 结果。

## 7. 风险自查

- 泄露风险：正式 `instruction.md` 只描述「动态图下 unique / unique_consecutive 对 0-size tensor 输入的行为异常」，不指出具体 `math.prod(x.shape) == 0` 分支逻辑或具体代码位置。
- 环境风险：低。任务为 Python-only，无需特殊镜像、外部服务或不可固定下载。
- flaky 风险：低。测试使用固定的 0-size tensor 构造，不依赖随机数差异或多设备同步。
- 拆分风险：低。该 PR 目标集中在 `manipulation.py` 中两个 API 的 0-size 早期返回，测试也明确指向 unique 和 unique_consecutive 的零尺寸分支，适合作为一个独立样本。
- 其他不确定点：完整任务包阶段应确认新增 F2P（`TestUniqueAPI_ZeroSize`、`TestUniqueConsecutive_ZeroSize`）在 `base_commit` 上确实失败，并选择同文件中已有的 `TestUniqueAPI`、`TestUniqueConsecutive` 等标准测试用例作为在 base 与修复后都稳定通过的 P2P nodeid。
