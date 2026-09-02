# 任务提案：PaddlePaddle__Paddle-79497

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79497`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79497
- PR 标题：`Use double precision for the norm_p value in norm reduce GPU kernel`
- `base_commit`：`31c7cfe559717accacf45dfc99d0042a43878bee`
- merged 时间：`2026-07-22T07:46:24Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：`wwaawwaaee`

## 2. 问题一句话

GPU 上的 `p_norm` 前向归约路径在处理 `float64` 输入且 `p` 为非整数时，会把 `norm_p` / `porder` 以 `float` 精度存储或传递，导致精度被截断，进而在严格容差下出现前向结果与参考实现不一致，并在集成的梯度回归检查中暴露连带误差。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：这是来自 Paddle 主仓已合入 PR 的真实数值精度问题，不是人为构造的 toy case。触发条件明确，来自 `float64` 输入、非整数 `p` 和 GPU reduction 路径的组合。
- **代表性**：样本覆盖 operator kernel、CUDA reduction dispatch、数值精度传播，以及通过集成 `check_grad` 体现的端到端一致性验证。它要求 agent 理解 `p_norm` 的前向归约链路如何影响最终数值表现，而不是只盯住单个 Python API 表面行为。
- **边界清楚**：目标边界是 GPU 路径下 `float64 + 非整数 p` 的精度传播，典型例子是 `p=2.3`。整数 `p`、普通 `float32`、零尺寸输入以及已有常规 `p_norm` 行为都应保持不变。Windows 上可能因为路径差异表现出不同的数值或执行路径，但这属于需要提示的不确定性，不是额外验收目标。
- **非平凡性**：任务不能被简化成“把某个 value 参数从 float 改成 double”。真正难点在于识别 `norm_p` / `porder` 在 GPU forward reduction 相关链路中的存储、传递和 dispatch 精度要求，并在不直接改 dedicated gradient kernel 的前提下，让严格 `float64` 参考检查和集成梯度回归都恢复通过。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cuda`
- 设备范围：`single_gpu`
- 模块标签：`[numerical_precision, reduction, p_norm, operator_kernel]`

## 5. 验证思路

- 目标测试文件 / 命令：在 `test/legacy_test/test_norm_all.py` 中补充 F2P，用 `float64` 输入和 `p=2.3` 覆盖严格前向结果检查，并保留 upstream 测试里的 `check_grad` 作为集成回归证据，同时覆盖 PIR / prim 相关路径；最小测试命令可收敛到该文件中的新增窄范围用例。
- 修复前预期：在 `base_commit` 上应用测试补丁后，已有整数 `p`、`float32`、零尺寸和普通 `p_norm` 用例仍通过，但新增 `float64 + p=2.3` 的严格 NumPy / reference 前向检查会失败；upstream 一并保留的 `check_grad` 也可作为集成回归证据暴露该数值问题。
- 修复后预期：继续应用代码补丁并在可用的 CUDA source build 环境中重新运行后，新增 F2P 通过，说明 GPU forward reduction 路径中的 `norm_p` / `porder` 精度传播已经与 `float64` 输入场景匹配；同时，集成的 `check_grad` 回归也应恢复通过，但这不意味着 PR 直接修改了独立的 gradient kernel。
- F2P 候选：`test/legacy_test/test_norm_all.py` 中新增 `float64` 输入、`p=2.3`、固定数据和固定 seed 的精确前向输出测试，并结合 upstream 已纳入的 `check_grad` 作为集成回归覆盖，同时覆盖 PIR / prim 路径。
- P2P 候选：沿用 `test/legacy_test/test_norm_all.py` 及同模块中已有的整数 `p`、`float32`、零尺寸输入和常规 `p_norm` 测试，作为回归护栏，确保修复不会破坏现有普通场景。

## 6. 环境与资源

- 是否能提供 Docker：无
- Dockerfile 或镜像地址：无
- Paddle 来源：source build
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不适用
- OS / Python / CUDA / cuDNN / 其他关键依赖：需要可重新编译并运行 Paddle CUDA kernel 的 source build 环境；proposal 阶段未承诺具体现成镜像，Windows 路径可能存在与 Linux 不同的执行或数值表现，需要在正式验证阶段单独确认。
- 硬件：单卡 GPU 即可
- patch 类型：含 CUDA kernel / operator 实现与 Python 测试
- 最小测试命令：建议收敛为 `test/legacy_test/test_norm_all.py` 中新增 F2P 与必要 P2P 的窄范围执行命令
- 是否有 oracle 日志：无

说明：

- 该任务依赖 CUDA source build，不能只靠未重编译的历史 wheel 验证 kernel 修复是否生效。
- 测试应使用固定输入数据和固定 seed，避免把数值波动误判为 flaky。
- 本 proposal 不做性能收益声明，验收重点是前向归约链路中的数值正确性，以及由 upstream `check_grad` 提供的集成一致性回归证据。

## 7. 风险自查

- 泄露风险：正式题面应只描述 `float64` 输入、非整数 `p`、GPU 前向归约路径下与参考实现不一致的可观察现象，以及 `check_grad` 可作为回归证据，不直接暴露具体 kernel 修改点或实现细节。
- 环境风险：任务需要 CUDA source build，环境准备成本高于纯 Python 任务；不同平台尤其是 Windows 可能存在路径差异或额外数值现象，这里只能作为风险提示，当前 proposal 不声称已完成运行验证。
- flaky 风险：主要风险来自浮点误差和不同执行路径，因此测试必须使用确定性固定数据、固定 seed，并以 NumPy / reference 结果为准，避免依赖随机输入。
- 拆分风险：低。该 PR 聚焦于同一类问题，即 `norm_p` / `porder` 在 GPU reduction 链路中的前向精度传播，以及其在集成回归中的可观察影响，适合作为单独样本。
- 其他不确定点：后续完整任务包阶段需要进一步确认 PIR / prim 覆盖在不同平台和构建选项下的最小稳定执行方式，并核实 Windows 相关差异是数值路径差异还是测试环境差异。
