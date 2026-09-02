# 任务提案：PaddlePaddle__Paddle-79510

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79510`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79510
- PR 标题：`[Operator Mechanism] Fix embedding invalid index handling`
- `base_commit`：`53909608869d33c42ad0479df7362d4b71e7f2da`
- merged 时间：`2026-07-21T01:53:07Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：`wwaawwaaee`

## 2. 问题一句话

这个 PR 的 benchmark 核心可以聚焦为 `embedding` 在 CPU 和 CUDA 前向路径上的安全契约：非法非 padding index 不能越界访问，且当词表 / weight 为空但 ids 非空时，必须在进入后端内存访问前直接报错。

更具体地说，所有非 padding 的 id 都必须落在 `[0, N)`；当权重第一维 `N=0` 且 ids 非空时，无论 CPU 还是 CUDA，都应在进入 backend memory access 前失败。空 ids 前向仍然合法，普通 backward 语义保持不变；上游 PR 中与 XPU 前向对齐、以及 empty ids dense gradient 清零相关的改动应作为非 gating 上下文保留，而不是纳入本 benchmark 的强制验收。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：问题来自已合入的真实 operator mechanism 修复，不是测试专用桩，而是 `embedding` 在边界输入下可能继续走向越界访问的真实安全性问题。
- **代表性**：它能代表 Paddle 在 `operator_kernel`、输入校验、零尺寸张量语义，以及 CPU / CUDA 前向错误路径一致性上的研发能力。
- **边界清楚**：benchmark 的强制目标只聚焦一条前向安全契约：非 padding id 必须满足 `[0, N)`；`N=0` 且 ids 非空必须失败，且失败点要早于后端内存访问。padding row、valid ids、empty ids forward、ordinary backward 都应保持既有正确行为。XPU 前向对齐与 empty ids dense gradient 清零属于上游 PR 的额外覆盖面，但不是本 proposal 的强制 gate。
- **非平凡性**：这不是单点 `if` 判断。即便只保留 CPU / CUDA gate，仍需要把普通非法 id、空 weight + 非空 ids、以及 `padding_idx` 边界一起梳理清楚，确保错误发生在正确层级，而不是等到 backend memory access 或 device 异常后才暴露。
- **Gold-extra-change caveat**：上游 gold patch 还包含 XPU 相关前向对齐和 empty ids dense gradient 清零修复，所以 benchmark 设计必须明确“上游改动范围”与“本任务强制验收范围”并不完全相同，避免 reviewer 误以为 proposal 在隐瞒 PR 的额外变更。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu` / `cuda`
- 设备范围：`single_gpu`
- 模块标签：`[embedding, operator_kernel, input_validation, zero_size]`

## 5. 验证思路

- 目标测试文件 / 命令：应围绕 `embedding` 的 CPU / CUDA 前向边界用例补充最小测试，并保留同文件或同模块中的正常行为回归。由于 PR 涉及编译后 kernel 与设备分支，验证应基于 Paddle source build，而不是只替换 Python 文件。
- 修复前预期：在 `base_commit` 上加入对应测试后，CPU 上普通非法 ids（尤其 `padding_idx=None` 与 `padding_idx=0` 两类边界）可能没有被一致拦截，CPU empty weight + nonempty ids 也可能缺少前置失败；CUDA 上 empty weight + nonempty ids 的错误可能在 backend memory access 之后才暴露，甚至表现为 device 侧异常。上游 PR 中还顺带修复了 XPU 对齐与 empty ids dense gradient 清零，但这些不作为本 benchmark 的 gating 失败条件。
- 修复后预期：应用 PR 的真实生产补丁并使用包含相应编译产物的构建后，CPU / CUDA 前向安全契约应成立：非 padding ids 一律受 `[0, N)` 约束；`N=0` 且 ids 非空时在进入后端访问前直接失败；valid ids、padding row、empty ids forward 和 ordinary backward 的既有语义不回归。XPU 相关改动可作为额外非 gating 证据记录，但不影响本任务通过判定。
- F2P 候选：
  - CPU invalid ids with `padding_idx=None`
  - CPU invalid ids with `padding_idx=0`
  - CPU empty weight + nonempty ids with `padding_idx=None`
  - CPU empty weight + nonempty ids with `padding_idx=0`
  - CUDA empty weight + nonempty ids
- P2P 候选：
  - valid ids
  - padding row
  - empty ids forward
  - ordinary embedding backward
- 验收说明：本 benchmark 的强制 acceptance 只要求 CPU + CUDA。CPU 负责覆盖 ordinary invalid ids、empty weight + nonempty ids 及 `padding_idx` 边界；CUDA 负责覆盖 empty weight + nonempty ids 的前向安全行为。XPU 前向 parity 和 XPU empty ids dense-gradient zeroing 可以作为非 gating 上下文、额外证据或后续 follow-up proposal，但不是本任务通过的必要条件。

## 6. 环境与资源

- 是否能提供 Docker：无
- Dockerfile 或镜像地址：暂无
- Paddle 来源：source build
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不适用。该任务依赖编译后的 operator kernel / device 实现，不能用未重编译的旧 wheel 完整验收。
- OS / Python / CUDA / cuDNN / 其他关键依赖：需要与 `base_commit` 兼容的 Paddle 源码构建环境；完整验证需具备 CPU 与单卡 CUDA 可运行环境。
- 硬件：CPU；单张 CUDA GPU。上游 PR 另含 XPU 改动，但不属于本 benchmark 的强制验收环境。
- patch 类型：含 C++ / kernel / 多后端设备实现，且需要编译产物参与验证。
- 最小测试命令：最终任务包应提供聚焦 `embedding` CPU / CUDA 前向边界契约的最小测试入口，并分别说明 CPU 与单卡 CUDA 的运行方式；proposal 阶段先明确 source build 是前提。
- 是否有 oracle 日志：无

说明：

- 该任务不适合宣称为纯 Python patch。即使测试补丁很窄，真实修复仍依赖编译后的 kernel 与设备分支。
- proposal 必须同时写清两件事：benchmark 的强制 gate 仅是 CPU / CUDA 前向安全契约；但上游 PR 的 gold patch 额外包含 XPU 相关修复，这些属于非 gating 范围，不能被故意省略。

## 7. 风险自查

- 泄露风险：proposal 只描述 `embedding` 的可观察契约和候选测试边界，没有暴露具体修改行、内部 helper 路径或 device kernel 细节。
- 环境风险：中等。完整验收仍依赖 source build 与单卡 CUDA，但不再把 XPU 作为门槛，因此明显低于 special_hardware 任务。
- flaky 风险：中低。目标测试主要是确定性的前向边界输入和错误路径；不同设备后端的报错形式、触发时机和构建差异仍可能带来非功能性噪声，因此断言应聚焦在“是否提前失败”“是否合法通过”这类稳定语义，而不是绑定具体报错文案。
- 拆分风险：低到中。当前 proposal 已把 mandatory gate 收敛到单一的 CPU / CUDA 前向安全契约，主 benchmark 不再需要再拆；但需要在任务说明里保留 gold patch 含 XPU 额外改动的 caveat，避免 verifier 或 reviewer 误解为任务与上游 PR 完全等宽。
- 其他不确定点：完整任务包阶段需要确认上游最终落地的测试入口，以及 CPU / CUDA 上最稳妥的前置失败断言方式；XPU 相关内容如果后续单独立项，再另行设计 follow-up 样本即可。
