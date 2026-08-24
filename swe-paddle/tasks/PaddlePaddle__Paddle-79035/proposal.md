# Task Proposal: PaddlePaddle__Paddle-79035

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79035`
- Issue 链接：无独立关联 issue；目标行为由 PR 描述和 review discussion 定义
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79035
- PR 标题：`[API Compatibility] Add aliases for apis in paddle.optimizer.lr`
- `base_commit`：`e55b609b31d6a00ab35d8fd6e651b2106319ba0d`
- merged 时间：`2026-05-22`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

`paddle.optim` 缺少 `lr_scheduler` 兼容 namespace，导致使用常见 scheduler 名称的代码无法直接通过该入口访问 Paddle 已有的 learning-rate scheduler。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源于已合入 PaddlePaddle 的 API Compatibility PR，目标是用户直接可见的公开 Python API。
- **代表性**：覆盖 framework API namespace 兼容、公开符号映射和 package import 暴露，这类兼容工作具有明确工程价值。
- **边界清楚**：production change 仅涉及 `python/paddle/optim/__init__.py` 与 `python/paddle/optim/lr_scheduler.py`；Gold commit 另更新一处上游 alias test。
- **非平凡性**：不仅要求模块存在，还要验证完整 alias contract 以及 `paddle.optim` package 初始化后 namespace 的可见性。
- **环境友好性**：可以用 installed Paddle wheel 提供依赖，并对 checkout production Python files 做 source overlay，无需 source build。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[optimizer, lr_scheduler, api_compatibility, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr79035_lr_scheduler_alias.py`
- 修复前预期：P2P 验证 legacy `paddle.optimizer.lr.StepDecay` 仍可构造并返回初始 learning rate；两个 F2P 分别因兼容模块缺失、`paddle.optim` 初始化后无 `lr_scheduler` namespace 而失败。
- 修复后预期：P2P 保持通过；兼容模块能加载，13 个公开兼容名称与 canonical scheduler 对象一致；`paddle.optim` package 初始化后可见 `lr_scheduler` namespace。
- P2P 候选：`test_p2p_legacy_step_decay_keeps_initial_learning_rate`

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用已安装的 Paddle wheel 提供 `paddle.optimizer.lr` runtime 依赖；测试通过 source overlay 执行 checkout 中目标 Python module/package 初始化逻辑，并用 controlled module doubles 隔离与目标无关的 sibling optimizer imports，避免依赖 wheel 是否已提供 `paddle.optim`，也无需编译 Paddle。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 仅描述公开 namespace、公开兼容名称和期望行为，不描述 Gold patch 的具体 import 写法。
- 环境风险：测试依赖可 import 的 Paddle wheel 与 pytest，但不依赖 GPU、NCCL、网络或 source build。
- flaky 风险：无并发、随机时序、训练或外部数据依赖，测试为确定性 import/API contract 检查。
- 拆分风险：Gold 同时新增兼容 module 并从 `paddle.optim` 暴露它；两项共同构成完整用户可观察 API，适合作为一个 task。
