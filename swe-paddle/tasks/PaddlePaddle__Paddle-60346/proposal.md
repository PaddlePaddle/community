# SWE-Paddle Proposal: PaddlePaddle/Paddle#60346

## 1. 任务来源

- Issue: `PaddlePaddle/Paddle#58067` — 新 IR Python API 适配升级
- PR: `PaddlePaddle/Paddle#60346`
- Title: `【PIR API adaptor No.272】Migrate LookAhead to pir`
- Base: `ac2de38a80dd6f2c66b4ef4ad515e209a0470fef`
- Gold: `caa171552152424a2adcfe6bef9babb2039434a2`

该 PR 已合入，并且最终 merge commit 相对其唯一父提交只修改 Python 文件：LookAhead、SGD 和对应单测，不需要重新编译 Paddle。

## 2. 任务目标

让 `paddle.incubate.optimizer.LookAhead` 在 PIR static graph 中具备与旧 static graph 对等的基本组网能力，包括：

- LookAhead 自身 global-step 状态的创建/更新；
- inner SGD optimizer 在 PIR Block 上创建 accumulator 的兼容性；
- PIR Value 形式 loss 传入 `LookAhead.minimize()`；
- legacy static 行为不回归。

## 3. SWE-bench 适配方式

Gold patch 仅包含两个 production Python 文件：

```text
python/paddle/incubate/optimizer/lookahead.py
python/paddle/optimizer/sgd.py
```

独立 test patch 新增 `test/swe_paddle/test_pr60346_lookahead_pir.py`。测试从仓库根目录读取上述官方源码路径，通过 AST 提取相关方法并执行真实 Python 控制流，因此不会因测试机安装的 Paddle wheel 版本不同而把验证错误地落到 site-packages 上。

不使用：

- 自定义 `PYTHONPATH`；
- GPU/CUDA；
- 网络或数据集下载；
- 多进程；
- Paddle source build。

## 4. P2P / F2P

### P2P

`test_legacy_increment_path_is_preserved`

验证旧 static graph 的 global-step 创建和 increment append-op 路径仍然存在。

### F2P

1. `test_pir_increment_uses_parameter_and_pir_increment`
   - Base 会落入 legacy global-var 路径并失败。
   - Gold 应在 PIR 路径完成状态创建和 increment。

2. `test_sgd_accepts_pir_block_for_accumulator_creation`
   - Base 只接受 legacy Block。
   - Gold 应接受 PIR Block。

3. `test_lookahead_minimize_accepts_pir_value`
   - Base 的 loss 类型约束拒绝 PIR Value。
   - Gold 应允许 PIR Value 并继续 inner optimizer/minimize 流程。

## 5. 难度评估

中等偏上。修复者需要同时理解 LookAhead 的状态管理、SGD accumulator 路径以及 legacy static/PIR 的类型差异；但无需进入 C++ kernel、编译系统或 SOT/bytecode 内部。

## 6. 风险

- deterministic：固定 fake runtime，不使用随机数。
- CPU-only：不触发 CUDA。
- source-bound：测试执行 checkout 中的 Python 方法，而不是检查字符串或依赖安装包里的同名实现。
- no implementation leakage：instruction 描述行为契约，不给出具体 Gold 代码。
