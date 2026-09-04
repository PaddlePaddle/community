# Task Proposal: PaddlePaddle__Paddle-78452

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78452`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78452
- PR 标题：`Support loading dataclass objects in paddle.load()`
- `base_commit`：`0156c9d3a222adaca16a394826654a9f449d11aa`
- Gold commit：`362b943a5a2823f9b2d4a2f0ffe2a2cff07789ab`
- merged 时间：`2026-03-25T12:48:48Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

修复 `paddle.load()` 无法恢复普通配置类和 dataclass 对象的问题，同时保持 restricted unpickler 对危险 pickle 行为的拦截。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle PR #78452，覆盖模型和训练配置随 checkpoint 保存后无法恢复的实际问题。
- **代表性**：模型文件通常同时包含 Tensor、NumPy 数据和 Python 配置对象，兼容性与安全性需要同时考虑。
- **边界清楚**：production change 只涉及模型对象遍历和 restricted unpickler 两个 Python 文件。
- **非平凡性**：实现既要允许安全的用户配置类，又不能放过具有危险 pickle hooks 的类，不能通过简单扩大白名单完成。
- **环境友好性**：来源 PR 测试使用内存缓冲区和临时文件，可在 CPU 环境稳定运行。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[framework_io, paddle_load, restricted_unpickler, dataclass, serialization]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_restricted_unpickler.py`
- 修复前预期：已有白名单数据和危险类拦截测试通过；来源 PR 新增的安全配置类/dataclass 加载测试失败；完整目标测试文件失败。
- 修复后预期：安全配置类和 dataclass 可以恢复，危险 `__reduce__` 等行为仍被拦截，完整目标测试文件通过。
- P2P 候选：包含 NumPy 数据的字典正常加载；`os.system` pickle 被拦截；带危险 `__reduce__` 的对象继续被拒绝。
- F2P 候选：安全 dataclass 被判定为可加载；安全 dataclass 和普通训练配置对象能够通过 restricted unpickler 恢复。
- 测试来源：`tests/test.patch` 基于来源 PR 的测试覆盖做 benchmark 适配；Gold-only `_is_safe_class` 不再在模块级导入，而是在相关 F2P 测试执行时延迟导入，使 Base 能完成测试收集，同时保持原有安全行为测试作为 P2P。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用可导入 Paddle 的 Python 环境；验证脚本把 checkout 中的目标 Python 行为覆盖到已安装 wheel，并在退出时恢复。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述可观察的加载结果与安全边界，没有给出类检查规则或实现步骤。
- 环境风险：历史 checkout 与当前 wheel 可能不完全一致；cross script 只覆盖目标函数和纯 Python restricted-unpickler 模块，并统一转换 Windows/Git Bash 路径。
- flaky 风险：测试不依赖网络、GPU、多进程或随机时序。
- 拆分风险：对象遍历与安全反序列化共同组成 `paddle.load()` 的目标流程，来源 PR 也将其作为一个问题处理。

