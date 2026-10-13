# 任务提案：PaddlePaddle__Paddle-79497

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79497`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79497
- PR 标题：`Use double precision for the norm_p value in norm reduce GPU kernel`
- `base_commit`：`31c7cfe559717accacf45dfc99d0042a43878bee`
- merged 时间：`2026-07-22`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：`wwaawwaaee`

## 2. 问题一句话

GPU `p_norm` 前向归约路径在 `float64` 输入且 `p` 为非整数时，`norm_p` / `porder` 以 `float` 精度存储或传递，导致前向结果与参考实现不一致，并在集成的 `check_grad` 回归中暴露连带误差。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自 Paddle 主仓已合入 PR 的真实数值精度问题，触发条件明确（`float64` 输入 + 非整数 `p` + GPU reduction）。
- **代表性**：覆盖 CUDA kernel、reduction dispatch、精度传播，以及通过集成 `check_grad` 体现的端到端一致性验证。
- **边界清楚**：目标边界是 GPU 路径下 `float64 + 非整数 p` 的精度传播（典型 `p=2.3`）；整数 `p`、`float32`、零尺寸输入及常规 `p_norm` 行为保持不变。
- **非平凡性**：难点在于识别 `norm_p` / `porder` 在 GPU forward reduction 链路中的存储、传递和 dispatch 精度要求，让严格 `float64` 参考检查和集成梯度回归恢复通过。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cuda`
- 设备范围：`single_gpu`
- 模块标签：`[numerical_precision, reduction, p_norm, operator_kernel]`

## 5. 验证思路

- 目标测试文件：`test/legacy_test/test_norm_all.py`，新增 F2P 并保留 upstream `check_grad` 作为集成回归证据，同时覆盖 PIR / prim 路径。
- 修复前预期：整数 `p`、`float32`、零尺寸和常规 `p_norm` 用例仍通过；新增 `float64 + p=2.3` 的严格 NumPy / reference 前向检查失败。
- 修复后预期：应用代码补丁并在 CUDA source build 环境重新运行后，新增 F2P 通过，集成的 `check_grad` 回归恢复通过。
- F2P 候选：`test_norm_all.py` 中新增 `float64` 输入、`p=2.3`、固定数据和固定 seed 的精确前向输出测试，严格 `1e-12` 容差、以 NumPy / reference 为准，覆盖 PIR / prim 路径。
- P2P 候选：同模块已有的整数 `p`、`float32`、零尺寸输入和常规 `p_norm` 测试，作为回归护栏。

## 6. 环境与资源

- 是否能提供 Docker：无
- Dockerfile 或镜像地址：无
- Paddle 来源：source build
- 如果使用 wheel，请填写 wheel URL、Python 版本和平台标签：不适用
- OS / Python / CUDA / cuDNN / 其他关键依赖：可重新编译并运行 Paddle CUDA kernel 的 source build 环境。
- 硬件：单卡 GPU 即可
- patch 类型：含 CUDA kernel / operator 实现与 Python 测试
- 最小测试命令：`test/legacy_test/test_norm_all.py` 中新增 F2P 与必要 P2P 的窄范围执行命令
- 是否有 oracle 日志：无
- 说明：需 CUDA source build 验证 kernel 修复；测试使用固定确定性数据；不做性能收益声明；`check_grad` 仅作为集成回归证据。

## 7. 风险自查

- 泄露风险：正式题面只描述 `float64` 输入、非整数 `p`、GPU 前向归约路径下与参考实现不一致的可观察现象，不暴露具体 kernel 修改点。
- 环境风险：任务需要 CUDA source build，环境准备成本高于纯 Python 任务；Windows 路径可能存在与 Linux 不同的执行或数值表现，仅作风险提示。
- flaky 风险：测试使用确定性固定数据、固定 seed，以 NumPy / reference 结果为准。
- 拆分风险：低。该 PR 聚焦 `norm_p` / `porder` 在 GPU reduction 链路中的前向精度传播，适合作为单独样本。
- 其他不确定点：后续需确认 PIR / prim 覆盖在不同平台和构建选项下的最小稳定执行方式。