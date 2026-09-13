# Task Proposal: PaddlePaddle__Paddle-78522

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78522`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78522
- PR 标题：`[Distributed] Replace \`os.system\` with \`os.kill\` in \`launch/main.py\``
- `base_commit`：`3493dbf5fdf1da8e59b8de87ec268ab386b9eefb`
- merged 时间：`2026-04-08T09:19:01Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

分布式 auto-tuner 在清理设备占用进程时，需要安全忽略无效或自身 PID，并在进程已退出或无权限时保持明确且不中断的清理语义。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源于已合入 Paddle develop 的真实分布式启动 PR。
- **代表性**：覆盖系统进程列表解析、跨平台信号选择和进程退出竞态等常见工程问题。
- **边界清楚**：production change 仅涉及 auto-tuner 调用点和通用 launch cleanup utilities。
- **非平凡性**：需要同时保持过滤、调用顺序、异常处理和平台差异，不能用简单字符串替换完成。
- **环境友好性**：upstream test 使用 mock，不启动真实训练、GPU、网络或外部服务。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, launch, auto-tuner, process-cleanup]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_launch_main_kill.py`
- 修复前预期：upstream test 中 PID 过滤、进程不存在处理和权限错误处理 3 个节点因目标契约缺失而失败。
- 修复后预期：上述 3 个 F2P 节点全部通过，完整目标测试文件通过。
- P2P：`test/legacy_test/test_launch_coverage.py::TestCoverage::test_find_free_ports`，这是 Base 已存在且 CPU 可稳定运行的 launch utility 回归节点，并由 `tests/test.sh` 实际执行。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：使用与 checkout 兼容的 Paddle wheel；该任务不需要 source build。
- 最小测试命令：`bash tests/test.sh`（实际执行 1 个既有 P2P + 3 个新增 F2P）
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 仅描述可观察清理契约，不给出 Gold helper 名称或具体修改步骤。
- 环境风险：历史 checkout 与安装 wheel 可能不一致，cross verifier 使用 checkout 中的 exact Python production files 作为 runtime carrier，并在退出时恢复。
- flaky 风险：F2P 使用 upstream mock test，不发送真实信号；P2P 仅使用本机回环端口和 CPU launch utility。
- 拆分风险：upstream test 直接覆盖清理 utilities，但不启动完整 auto-tuner；因此对 `launch/main.py` 集成路径的直接 runtime 覆盖有限，cross verifier 额外校验两个 production file 的 exact Gold blob。
