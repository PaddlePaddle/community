# Task Proposal: PaddlePaddle__Paddle-79369

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-79369`
* PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79369
* PR 标题：`【BugFix】Fix bug of check_memory_usage`
* `base_commit`：`199073cd2021dd05efd8b0fe79797b838f68df41`
* 任务类型：`bug_fix`
* 后续联系人：TBD

## 2. 问题一句话

修复 Fleet `check_memory_usage` 调用不支持的 CPU 内存接口时直接报错的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：来自 Paddle 主仓已经合入的真实 bug。
* **边界清楚**：生产代码修改集中在一个 Python 工具函数，不涉及其他训练逻辑。
* **问题明确**：CPU 内存接口不支持时，`check_memory_usage` 会抛出异常，导致内存日志记录中断。
* **验证成本低**：通过 mock 即可复现，不需要 GPU、分布式多进程、外部服务或 C++ 编译。

## 4. 任务类型和标签

* 任务类型：`bug_fix`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[distributed, fleet, logging, memory, python]`

## 5. 验证思路

* 目标命令：`bash tests/test.sh`
* 目标文件：`test/legacy_test/test_check_memory_usage.py`
* P2P：验证现有的设备内存日志和 `free -h` 系统内存日志仍能正常记录。
* F2P：模拟 CPU 内存接口存在但调用时抛出异常，验证 `check_memory_usage` 能继续执行，不会因为该接口中断。
* 修复前：P2P 通过，F2P 因 `RuntimeError: unsupported CPU memory API` 失败。
* 修复后：P2P 与 F2P 均通过。([GitHub][2])

## 6. 环境与资源

* 资源需求：CPU
* GPU、分布式多进程、网络服务和外部模型：不需要
* patch 类型：Python 工具函数修改 + Python legacy test
* 构建要求：Python-only patch，无需重新编译 C++ core

## 7. 风险自查

* 泄露风险：任务说明只描述报错场景和预期结果，不给出具体代码修改方式。
* 环境风险：设备接口和系统命令均通过 mock 隔离，不依赖运行机器的实际内存环境。
* flaky 风险：测试只检查确定的 mock 调用、异常和日志内容，不包含随机行为。
