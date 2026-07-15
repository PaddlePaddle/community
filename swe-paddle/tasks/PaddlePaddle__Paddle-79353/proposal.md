# Task Proposal: PaddlePaddle__Paddle-79353

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-79353`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/79353
- PR 标题：`[Bug Fix] Fix p2p local_var bug`
- `base_commit`：`406c7afec699c23158e7ff62a0f1afb306e72afe`
- `gold_commit`：`15c873afa5dd01faeddbd39f6d985a69926c384e`
- merged 时间：`2026-06-23`
- 后续联系人：TBD

## 2. 问题一句话

完善 Pipeline P2P overlap 模式在 stage 边界无通信路径下的返回行为。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：任务来自已合入的 Paddle 分布式训练 bug fix。
- **边界清楚**：生产代码仅涉及一个 Python 文件中的两个边界分支。
- **可复现性**：测试可直接调用通信辅助接口，不需要启动多进程或真实通信后端。
- **区分度**：修复需要理解 overlap 模式的返回约定，而不是仅规避异常。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, pipeline_parallel, p2p, overlap, python]`

## 5. 验证思路

- 目标测试命令：`bash tests/test.sh`
- 目标测试文件：`test/legacy_test/test_p2p_overlap_boundary.py`
- 修复前预期：非 overlap 回归用例通过；两个 overlap stage 边界用例失败。
- 修复后预期：全部目标测试通过。

## 6. 环境与资源

- 资源需求：CPU
- Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
- patch 类型：Python-only
- 最小测试命令：`bash tests/test.sh`

## 7. 风险自查

- 泄露风险：任务说明只描述接口行为和验收标准，不指出具体修改位置。
- 环境风险：无需 GPU、NCCL 或多进程分布式环境。
- flaky 风险：测试不执行真实通信，结果应稳定。
