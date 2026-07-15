# Task Proposal: PaddlePaddle__Paddle-78922

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-78922`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/78922
- PR 标题：`[FlexCheckPoint] fix memory leaking of a recursive function`
- `base_commit`：`c55db2546c87e19fc78384c3497383298f3e2375`
- merged 时间：2026-05-11
- 你的身份：熟悉该模块的 contributor
- 后续联系人：TBD

## 2. 问题一句话

完善 `flatten_state_dict` 的对象生命周期管理，避免函数返回后继续持有已无外部用途的 Tensor 引用。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 Paddle FlexCheckpoint bug-fix PR。
- **边界清楚**：生产修改集中在一个 Python 函数，目标行为可通过确定性的引用计数测试验证。
- **可复现性**：CPU-only，不依赖 GPU 显存、系统 RSS、分布式进程或外部服务。
- **回归护栏**：同时验证扁平化 key、mapping 和反向还原行为未发生变化。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 执行后端：`cpu`
- 设备范围：`cpu_only`
- 模块标签：`[distributed, flex_checkpoint, state_dict, python_gc]`

## 5. 验证思路

- 目标命令：`bash tests/test.sh`
- 测试文件：`test/legacy_test/test_flatten_state_dict_lifetime.py`
- F2P：关闭周期 GC 后删除返回结果，验证输入 Tensor 的额外引用是否立即释放。
- P2P：验证嵌套 state dict 的 flatten、mapping 和 unflatten 行为保持正确。
- 修复前预期：P2P 通过，F2P 失败。
- 修复后预期：P2P 与 F2P 均通过。

## 6. 环境与资源

- Python 3
- Paddle Python runtime
- CPU
- `pytest`
