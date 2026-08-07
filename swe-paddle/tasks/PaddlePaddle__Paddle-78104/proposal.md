# Task Proposal: PaddlePaddle__Paddle-78104

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78104`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78104
- PR 标题：`[Bugfix] paddle.cuda.device(tensor.place)`
- `base_commit`：`d85ad0fca9513ff7d1f0a552649f9136e94cf2a5`
- merged 时间：`2026-03-03`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 device context 接收 Tensor 返回的 generic `Place` 时丢失真实 device type 或 device id 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle user-experience BugFix PR。
- **代表性**：覆盖 Tensor place、device context、generic wrapper 与 concrete Place conversion。
- **边界清楚**：上游只修改一个 production file 和一个 unit-test file，solution 仅保留 production change。
- **非平凡性**：修复不仅要识别 generic Place，还要分别保留 CPU、CUDA、XPU 和 custom device 的类型与 id。
- **区分度信号**：该任务能够区分仅对某一种设备做特判的局部修复，与完整覆盖 generic Place conversion 且不破坏已有输入行为的修复。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[device, place, context_manager, cuda, xpu, custom_device, python_api]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_device_place_conversion_contract.py`
- 修复前预期：string device 与 concrete Place P2P 应 pass；generic Place conversion F2P 应 fail。
- 修复后预期：应用 production-only Gold patch 后，P2P 与 F2P 均应 pass。
- P2P 候选：`"cpu"` string conversion，以及 concrete CUDAPlace passthrough。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only production change
- 环境建议：通过 AST 加载 checkout 中的 `_convert_to_place`，使用 controlled core Place doubles 验证 device type 与 id，不依赖实际 GPU/XPU/custom hardware
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述 observable conversion contract，不指出具体条件表达式。
- 环境风险：不依赖 Paddle wheel、GPU、XPU、custom device runtime 或 source build。
- flaky 风险：所有 Place doubles 和 device ids 固定，测试无随机性与外部依赖。
- 拆分风险：所有场景都属于 `_convert_to_place` 的同一 conversion contract。
