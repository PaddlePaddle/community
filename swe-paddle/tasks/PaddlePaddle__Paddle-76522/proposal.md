# Task Proposal: PaddlePaddle__Paddle-76522

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-76522`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/76522
- PR 标题：`[Compat] Auto register compat module overrides when enable torch proxy`
- `base_commit`：`b5efb98a163a2be2505e72266841e64b88254a8a`
- Gold commit：`20d9626540daf86096cc5bd11c9b84b398ce7138`
- merged 时间：`2025-11-24`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

完善 Torch Proxy 的兼容接口覆盖机制，使其在启用时自动注册 `paddle.compat` 中公开提供的接口，并确保嵌套模块中的兼容实现能够通过对应的 `torch` 命名空间正确访问。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle PR #76522，不是人工构造的需求。
- **类型多样性**：该任务属于 compatibility layer 的功能完善，可补充 SWE-Paddle 中以常规 bug fix 为主的任务分布。
- **可观察性**：兼容接口是否被自动注册、嵌套模块是否返回预期实现，以及未覆盖属性是否保持原有代理行为，都可以通过运行时对象身份稳定验证。
- **边界清楚**：production change 集中在 `python/paddle/compat/proxy.py`，原 PR 中的测试改动可以与 Gold production patch 清晰分离。
- **环境友好性**：核心逻辑涉及 Python 模块导入、属性访问和代理行为，可以在 CPU 环境中直接运行 checkout 源码，无需 GPU 或 Paddle native source build。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[compat, torch_proxy, import_system, module_proxy, api_compatibility]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr76522_torch_proxy_compat_overrides.py`
- 修复前预期：没有兼容接口覆盖的属性应继续按原有逻辑返回 Paddle 模块中的对应对象；自动注册 `paddle.compat` 公开接口以及嵌套模块覆盖相关测试应失败。
- 修复后预期：应用 production Gold patch 后，P2P 与全部 F2P 均应 pass；公开兼容接口能够通过属性访问和 import 语句从对应的 `torch` 命名空间获得。
- P2P 候选：没有兼容实现的属性继续返回原始 Paddle 对象；已有的 Torch Proxy override 行为保持不变。
- F2P 候选：启用 proxy 时自动注册 `paddle.compat` 中公开导出的接口；嵌套模块中的兼容接口能够通过父模块属性访问和直接 import 正确生效；未公开接口不会被注册。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述兼容接口自动生效、嵌套模块访问和默认代理行为等运行时结果，不透露 Gold patch 中的类设计、注册过程或内部数据结构。
- 环境风险：测试不依赖真实 PyTorch 安装，也不要求历史 Paddle wheel 与 source checkout 完全匹配。
- flaky 风险：测试使用受控的 Python 模块和确定性的对象身份断言，不依赖网络、GPU、多进程或随机行为。
- 拆分风险：兼容接口自动注册和嵌套模块覆盖共同构成 Torch Proxy 兼容行为的完整改进，且集中在同一 production 文件中，适合作为单个任务。
