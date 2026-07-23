# Task Proposal: PaddlePaddle__Paddle-75274

## 1. 来源信息

- Instance ID：`PaddlePaddle__Paddle-75274`
- PR 链接：https://github.com/PaddlePaddle/Paddle/pull/75274
- PR 标题：`【UnitTestFix No.10】fix test_normal.py`
- `base_commit`：`5d1846ae16a8cecad8545b83d53a56e1a0eebe73`
- merged 时间：`2025-09-22`
- 你的身份：原 PR 作者
- 后续联系人：@Echo-Nie

## 2. 问题一句话

修复 `normal` 单元测试对全局静态图/动态图状态的隐式依赖，避免测试顺序变化后在错误模式下构图或创建静态输入。

## 3. 为什么适合作为 SWE-Paddle 样本

- **真实性**：来自已合入的 Paddle bug-fix PR。
- **目标明确**：本任务修复的工程对象就是单元测试实现，gold patch 修改测试文件符合原始 PR 目标。
- **范围可控**：上游仅修改 `test/legacy_test/test_normal.py` 一个文件。
- **独立判分**：`tests/test.patch` 新增独立 verifier，不覆盖 gold patch 的修改区域。
- **行为验证**：verifier 实际执行目标测试类的静态和动态图路径，不做源码字符串匹配。
- **确定性**：Base F2P 由执行模式错误触发；不依赖 GPU、网络、随机统计阈值或外部数据。
- **回归保护**：P2P 验证未修改的动态图路径继续工作。

## 4. 任务类型和标签

- 任务类型：`bug_fix`
- 子类型：`unit_test_maintenance`
- 执行后端：`cpu`
- 模块标签：`[normal, unittest, static_graph, dygraph, test_isolation]`

## 5. 验证设计

目标命令：

```bash
bash tests/test.sh
```

验证项：

1. **P2P**：缩短采样次数后执行既有动态图辅助路径，确认输出形状正确。
2. **F2P - scalar**：从动态图模式调用标量参数的静态辅助路径。
3. **F2P - tensor**：从动态图模式调用 Tensor 参数的静态辅助路径。
4. **Solution**：应用 gold patch 后重复上述验证，并检查目标文件 blob 与 Gold commit 完全一致。

## 6. 环境与资源

- 资源需求：CPU
- Python 依赖：PaddlePaddle、NumPy、pytest
- Paddle 重新编译：不需要
- GPU、分布式运行、网络服务和数据集：不需要

## 7. 风险自查

- **答案泄露**：题面描述模式隔离要求，但不指定具体 API 调用位置、代码缩进结构或返回路径。
- **测试重叠**：新增 verifier 与目标文件分离，不改写 gold patch 的目标代码。
- **性能风险**：verifier 将原测试的采样次数降至 2，只检查执行路径和形状，不运行耗时统计测试。
- **环境风险**：CPU helper 替代设备探测，仅用于独立 verifier，避免依赖本机 GPU。
- **投机风险**：测试同时覆盖标量和 Tensor 静态路径，并保留动态图 P2P。
