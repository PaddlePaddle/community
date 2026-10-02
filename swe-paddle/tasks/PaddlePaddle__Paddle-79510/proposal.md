# 任务提案：PaddlePaddle__Paddle-79510

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79510`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79510
- PR 标题：`[Operator Mechanism] Fix embedding invalid index handling`
- `base_commit`：`53909608869d33c42ad0479df7362d4b71e7f2da`
- merged 时间：`2026-07-21`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：`wwaawwaaee`

## 2. 问题一句话

修复 `embedding` 前向的 index 安全契约：所有非 padding id 必须落在 `[0, N)`，且当 weight 第一维 `N=0` 而 ids 非空时，CPU / CUDA 都必须在进入后端内存访问前直接报错。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的真实 operator mechanism 修复，不是合成任务。
- **代表性**：覆盖 `embedding` 的输入校验、零尺寸张量语义和 CPU / CUDA 前向错误路径一致性。
- **边界清楚**：强制 gate 只聚焦 CPU / CUDA 前向安全契约；padding row、valid ids、empty ids forward、ordinary backward 保持既有行为。
- **非平凡性**：需要把普通非法 id、empty weight + nonempty ids、`padding_idx` 边界一起梳理，确保错误发生在正确层级而非 backend memory access 或 device 异常之后。
- **区分度信号**：能区分只修单点判断的局部修复，与在 CPU / CUDA 前向各边界下都保持安全契约的完整修复。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu` / `cuda`
- 设备范围：`single_gpu`
- 模块标签：`[embedding, operator_kernel, input_validation, zero_size]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：围绕 `embedding` CPU / CUDA 前向边界的测试（任务包阶段确认入口）
- 修复前预期：CPU invalid ids（`padding_idx=None` / `0`）、CPU empty weight + nonempty ids（`padding_idx=None` / `0`）、CUDA empty weight + nonempty ids 应 fail。
- 修复后预期：应用 `solution/code.patch` 后，F2P 与 P2P 均应 pass。
- P2P 候选：valid ids、padding row、empty ids forward、ordinary embedding backward。

## 6. 环境与资源

- 资源需求：CPU；单张 CUDA GPU
- Paddle 来源：`PaddlePaddle/Paddle` source build at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：C++ / kernel / 多后端设备实现，需编译产物参与验证
- 环境建议：需与 `base_commit` 兼容的源码构建环境；gold patch 另含 XPU 前向对齐与 empty ids dense gradient 清零改动，均非本任务强制验收范围
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：无

## 7. 风险自查

- 泄露风险：只描述 `embedding` 的可观察契约和候选测试边界，不指出具体修改行或 kernel 细节。
- 环境风险：中等。依赖 source build 与单卡 CUDA，但不把 XPU 作为门槛。
- flaky 风险：中低。断言聚焦"是否提前失败 / 是否合法通过"的稳定语义，不绑定具体报错文案。
- 拆分风险：低。强制 gate 已收敛到单一 CPU / CUDA 前向安全契约；gold patch 含 XPU 额外改动，作为非 gating 上下文说明即可。