# Task Proposal: PaddlePaddle__Paddle-76522

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-76522`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/76522
- PR 标题：`[Compat] Auto register compat module overrides when enable torch proxy`
- `base_commit`：`b5efb98a163a2be2505e72266841e64b88254a8a`
- Gold commit：`20d9626540daf86096cc5bd11c9b84b398ce7138`
- merged 时间：`2025-11-24`
- 你的身份：原 PR 作者 / reviewer / 熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

重构 torch proxy override 机制，使启用 proxy 时自动注册 `paddle.compat` 的公开兼容接口，并让嵌套模块中的 override 能通过父级 proxy 正确生效。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle PR #76522，不是人工构造的需求。
- **类型多样性**：任务以 compatibility-layer refactor 为主，可补充 SWE-Paddle 中以 bug fix 为主的任务分布。
- **可观察性**：是否自动注册 compat API、嵌套模块是否返回 override、无 override 时是否 fallback 都可以通过运行期对象身份直接验证。
- **边界清楚**：production change 集中在 `python/paddle/compat/proxy.py`，原 PR test change 可与 Gold production patch 分离。
- **环境友好性**：核心逻辑为 Python import/proxy 行为，可在 CPU 环境通过 checkout 源码直接执行，无需 GPU 或 Paddle native source build。

## 4. 任务类型和标签

- 任务类型：`refactor`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[compat, torch_proxy, import_system, module_proxy, refactor]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr76522_torch_proxy_compat_overrides.py`
- 修复前预期：没有 override 的属性 fallback 测试应 pass；自动 compat 注册和嵌套模块 override 测试应 fail。
- 修复后预期：应用 production Gold patch 后，P2P 与全部 F2P 均应 pass。
- P2P 候选：`ProxyModule` 对无 override 属性继续返回原始模块属性。
- F2P 候选：启用 proxy 时自动发现公开 compat export；嵌套子模块中的 override 通过父模块 proxy 生效。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用可运行 `pytest` 的 Python 环境，直接执行 checkout 中 `python/paddle/compat/proxy.py` 的真实控制流，并用受控 fake modules 隔离 import side effects；无需 Paddle source build
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：`instruction.md` 只描述自动注册、嵌套 override 和 fallback 的用户可观察行为，不提供 Gold 的类设计或具体实现步骤。
- 环境风险：测试不依赖真实 PyTorch 安装，也不要求历史 Paddle wheel 与 checkout 完全匹配。
- flaky 风险：测试使用受控 Python modules 和对象身份断言，不依赖网络、GPU、多进程或随机行为。
- 拆分风险：自动 compat 注册和嵌套模块 override 是同一 torch proxy compatibility refactor 的两个必要部分，集中在同一 production 文件中，适合作为单个任务。
