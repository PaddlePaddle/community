# Task Proposal: PaddlePaddle__Paddle-40111

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-40111`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/40111
- PR 标题：`add profiler statistic helper`
- `base_commit`：`10325a82e1032c3397b6f6611f558eb18ede0b07`
- merged 时间：`2022-03-08T01:55:55Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

为 profiler 增加可靠的时间区间归并、交集和差集计算，避免统计重叠事件时重复计时或漏计。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来源 PR 为 profiler 性能统计补充了实际需要的时间区间运算能力。
- **代表性**：覆盖排序、重叠、包含、相邻、空输入以及两个区间集合之间的组合运算。
- **边界清楚**：production change 仅新增一个 225 行 Python helper，不涉及算子、模型加载或图执行。
- **非平凡性**：多个双指针分支需要保持有序输出并正确处理边界，无法靠单点特判完成。
- **环境友好性**：来源 PR 的 137 行单测保持 exact blob；测试直接加载 checkout 文件，可在纯 CPU 环境运行。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[profiler, statistics, intervals, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_paddle/test_pr40111_profiler_ranges.py`
- 修复前预期：现有 profiler scheduler 的 P2P 通过；来源 PR 的 merge、intersection 与 subtract 用例因区间工具缺失而失败。
- 修复后预期：P2P 和来源 PR 的全部 10 个区间测试通过；`tests/test.sh` 通过 adapter 运行 exact upstream test file。
- P2P 候选：现有 profiler scheduler 在 closed、ready、record 和 record-and-return 状态之间的切换顺序保持不变。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：无需 source build 或 Paddle wheel；adapter 直接执行 checkout 中的 Python 文件和 exact upstream tests。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：instruction 只说明所需区间行为，不描述 Gold patch 的循环结构或局部变量。
- 环境风险：测试不访问网络，不加载模型，也不依赖 GPU、算子或动态图/静态图。
- flaky 风险：所有输入均为确定性的整数区间，没有计时、并发或随机因素。
- 拆分风险：来源 PR 只新增一个 production helper 和对应单测，专注于同一统计问题。
