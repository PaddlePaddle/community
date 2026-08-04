主要需要把 `static graph` 统一改为原 PR 使用的 `static mode`，同时删掉 `contract`、`propagation`、`unknown dimension` 等生硬表达。原 PR 只修改了 `python/paddle/tensor/linalg.py` 和 `paddle/phi/infermeta/binary.cc`：Python 侧放宽对 `-1` 维度的检查，C++ 侧在一边为 `-1` 时使用另一边已知的维度推导结果。([GitHub][1])

# Task Proposal: PaddlePaddle__Paddle-56135

## 1. 来源信息

* Instance ID：`PaddlePaddle__Paddle-56135`
* PR 链接：[https://github.com/PaddlePaddle/Paddle/pull/56135](https://github.com/PaddlePaddle/Paddle/pull/56135)
* PR 标题：`[BugFix] fix bmm op bugs in static mode with dynamic shape`
* `base_commit`：`4f2cf7fbcaca52bb9625dc6be944f552ea1d71d5`
* merged 时间：`2023-08-16`
* 你的身份：熟悉该模块的 contributor
* 后续联系人：TBD

## 2. 问题一句话

修复 `paddle.bmm` 在 `static mode` 下处理包含 `-1` 的 shape 时，错误判断输入维度不匹配或无法正确推导输出 shape 的问题。

## 3. 为什么适合作为 SWE-Paddle 样本

* **真实性**：该任务来自已合入的 Paddle BugFix PR，问题由 PaddleSOT 使用场景发现，不是人工构造的需求。([GitHub][1])
* **代表性**：该任务同时涉及 Python API 的输入检查和 C++ 侧的输出 shape 推导，能够覆盖 `static mode` 下常见的 dynamic shape 问题。
* **边界清楚**：production change 只涉及 `python/paddle/tensor/linalg.py` 和 `paddle/phi/infermeta/binary.cc`，修改范围集中。([GitHub][2])
* **非平凡性**：仅放宽 Python 侧的维度检查并不能完整修复问题。C++ 侧还需要在一边为 `-1`、另一边为已知值时，正确检查维度并推导输出 shape。

## 4. 任务类型和标签

* 任务类型：`bug_fix`
* 执行后端：`cpu`
* 设备范围：`cpu_only`
* 模块标签：`[bmm, static_mode, dynamic_shape, infermeta, python_api, cpp]`

## 5. 验证思路

* 目标测试命令：`bash tests/test.sh`
* 目标测试文件：`test/legacy_test/test_bmm_dynamic_shape_contract.py`
* 修复前预期：已知且兼容的 shape 测试，以及已知维度不匹配时报错的测试应通过；包含 `-1` 的 Python API 检查和 C++ shape 推导测试应失败。
* 修复后预期：继续应用 `solution/code.patch` 后，P2P 与全部 F2P 均应通过。
* P2P 候选：两个输入的 shape 均已知且兼容时，输出 shape 保持正确；batch 维度或矩阵相乘维度均已知且不相等时，仍应正常报错。
* F2P 候选：batch 维度只有一边为 `-1` 时允许构建程序，并使用另一边的已知值推导输出 shape；矩阵相乘维度只有一边为 `-1` 时不应被错误判定为不匹配。

## 6. 环境与资源

* 资源需求：CPU
* Paddle 来源：`PaddlePaddle/Paddle` source checkout at `base_commit`
* 是否能提供 Docker：暂无
* patch 类型：Python + C++ infermeta
* 最小测试命令：`bash tests/test.sh`
* 是否有 oracle 日志：由 SWE-Paddle verifier 结果另行维护

## 7. 风险自查

* 泄露风险：正式 `instruction.md` 只描述一边为 `-1` 时的正确行为，不给出具体判断条件或内部辅助函数的实现。
* 环境风险：测试需要可用的 C++17 编译器，但不需要完整编译 Paddle，也不依赖 CUDA 或 GPU。
* flaky 风险：测试使用固定 shape，不涉及随机数、并发或外部数据。
* 拆分风险：Python 侧的输入检查和 C++ 侧的 shape 推导共同决定 `paddle.bmm` 对 dynamic shape 的处理，属于同一个问题，不适合拆成两个样本。

[1]: https://github.com/PaddlePaddle/Paddle/pull/56135 "[BugFix] fix bmm op bugs in static mode with dynamic shape by 2742195759 · Pull Request #56135 · PaddlePaddle/Paddle · GitHub"
[2]: https://github.com/PaddlePaddle/Paddle/pull/56135/changes "[BugFix] fix bmm op bugs in static mode with dynamic shape by 2742195759 · Pull Request #56135 · PaddlePaddle/Paddle · GitHub"
