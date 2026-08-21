# Task Proposal: PaddlePaddle__Paddle-60417

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-60417`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/60417
- PR 标题：`[auto config] Resume from history csv file`
- `base_commit`：`e4b39bb56a4e55213383e96daf262f4f72c1811d`
- merged 时间：`2023-12-29T07:40:45Z`
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

auto-tuner 重新启动后无法复用历史 CSV 中已经完成的配置，会重复运行全部调优任务。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：长时间分布式调优可能被中断，恢复后重复运行已完成任务会直接增加训练成本。
- **代表性**：任务同时涉及历史数据读取、配置匹配、结果记录和 launch 调度，是典型的断点恢复问题。
- **边界清楚**：修改只服务于 auto-tuner 从历史 CSV 恢复，不改变训练脚本、搜索算法或分布式执行后端。
- **非平凡性**：实现需要恢复 CSV 中的真实数据类型，正确识别同一配置，并在不启动控制器的情况下把历史结果接回现有记录与调度流程。
- **环境友好性**：测试使用受控的 CPU doubles 隔离真实训练进程，但执行 checkout 中的 `AutoTuner` 和完整 `launch()` 控制流。

## 4. 任务类型和标签

- 任务类型：`feature_enhancement`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, auto-tuner, launch, resume]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/swe_test/test_auto_tuner_resume.py`
- 修复前预期：既有搜索与 history list 行为通过；历史 CSV 读取、配置匹配和 launch 跳过已完成任务三个目标场景失败。
- 修复后预期：CSV 值恢复为可用类型并生成副本，配置按历史顺序匹配，命中的任务写入 recorder 且不启动训练控制器；既有搜索行为保持通过。
- P2P 候选：`test_existing_search_and_history_flow_is_unchanged`。
- F2P 候选：`test_resume_history_loads_values_and_preserves_a_copy`、`test_resume_lookup_returns_first_matching_configuration`、`test_launch_reuses_history_without_starting_training`。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- 是否能提供 Docker：暂无
- patch 类型：Python-only
- 环境建议：测试通过 `importlib` 直接执行 checkout 中的目标模块，并使用 controlled doubles 隔离搜索器、记录器和训练控制器；无需安装或覆盖 Paddle wheel，也无需启动分布式进程。
- 最小测试命令：`bash tests/test.sh`
- 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

- 泄露风险：任务说明只描述中断恢复的用户行为，没有给出 Gold patch 的方法名、分支位置或具体代码结构。
- 环境风险：测试只依赖 Python 标准库和 pytest，直接读取 checkout 源码，不受已安装 Paddle wheel 版本影响。
- flaky 风险：不启动训练进程、不访问 GPU、网络或外部数据集，历史文件全部在 pytest 临时目录中生成。
- 拆分风险：CSV 恢复、配置匹配和跳过重复训练共同构成同一个断点恢复流程，来源 PR 没有混入其他问题。
