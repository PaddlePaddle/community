# Task Proposal: PaddlePaddle__Paddle-79369

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79369`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79369
- PR 标题：`【BugFix】Fix bug of check_memory_usage`
- `base_commit`：`199073cd2021dd05efd8b0fe79797b838f68df41`
- 任务类型：`bug_fix`
- 后续联系人：TBD

## 2. 问题一句话

完善 Fleet `check_memory_usage` 在 CPU 运行环境中的内存日志行为。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自 Paddle 主仓已合入的真实 bug-fix。
- **边界清楚**：生产代码修改集中在一个 Python 工具函数。
- **行为明确**：修复前目标场景稳定失败，修复后正常完成。
- **验证成本低**：不依赖 GPU、分布式多进程、外部服务或 C++ 编译。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, fleet, logging, memory, python]`

## 5. 验证思路

- 目标命令：`bash tests/test.sh`
- 目标文件：`test/legacy_test/test_check_memory_usage.py`
- P2P：验证已有设备和系统内存日志流程保持可用。
- F2P：验证 CPU 运行环境下的目标日志场景能够正常完成。
- 修复前：P2P 通过，F2P 失败。
- 修复后：P2P 与 F2P 均通过。

## 6. 环境与资源

- 资源需求：CPU
- GPU、分布式多进程、网络服务和外部模型：不需要
- patch 类型：Python 工具函数修改 + Python legacy test
- 构建要求：Python-only patch，无需重新编译 C++ core

## 7. 风险自查

- 泄露风险：任务说明只描述目标行为和验收标准，不暴露具体修改位置。
- 环境风险：设备接口和系统命令由测试 mock 隔离。
- flaky 风险：断言只依赖确定性的 mock 调用和日志内容。
