# Task Proposal: PaddlePaddle__Paddle-18687

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-18687`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/18687
- PR 标题：`add parameter server launch`
- `base_commit`：`d07ad4c6059db28c5f384a25190385742d9ba718`
- merged 时间：`2019-07-22T14:11:50Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为本地参数服务器训练提供统一启动入口，根据 server/worker 数量生成一致的进程环境并管理完整任务生命周期。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源 PR 解决本地启动参数服务器任务时需要手工组织多个进程的问题。
- **代表性**：覆盖命令行解析、角色环境生成、endpoint 编排、子进程启动和失败传播。
- **边界清楚**：production change 仅新增一个 Python launcher，不涉及算子、模型权重或 GPU 执行。
- **非平凡性**：需要同时保证两类角色的数量、编号、共享配置、参数转发与进程退出语义。
- **环境友好性**：测试通过 controlled `Popen` doubles 验证真实控制流，可在 CPU 环境稳定运行。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, parameter_server, launcher, multiprocessing]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr18687_launch_ps.py`
- 修复前预期：现有 collective launcher 的参数转发测试通过；参数服务器 launcher 不存在，因此参数解析、角色编排和失败传播用例失败。
- 修复后预期：参数服务器命令行、server/worker 环境、命令参数、等待行为及非零退出码传播全部通过。
- P2P 候选：现有 `paddle.distributed.launch` 仍能把训练脚本及其余参数完整解析出来。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：无需 source build；使用 Python、pytest 和 controlled subprocess doubles 即可运行。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只描述启动器的外部行为，不给出 Gold patch 的具体代码结构。
- 环境风险：不启动真实训练进程，不依赖 Paddle wheel、网络、外部服务或 GPU。
- flaky 风险：进程行为由确定性的 doubles 记录，不依赖端口竞争和真实多进程时序。
- 拆分风险：来源 PR 只有一个 production file，所有验证围绕同一个参数服务器启动入口。
