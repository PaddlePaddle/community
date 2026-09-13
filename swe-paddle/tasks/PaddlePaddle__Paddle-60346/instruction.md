# Make LookAhead optimizer work in PIR static programs

## 详细描述

`paddle.incubate.optimizer.LookAhead` 在旧 static graph 中可以包装一个 inner optimizer，并维护 slow parameters、全局 step 以及每 `k` 步一次的 LookAhead 更新。但是在 PIR static graph 中，同样的优化器流程目前不能正常组网：与 program/block、持久化 step 状态和 loss 类型相关的旧 static graph 假设会阻止 PIR program 使用 LookAhead。

修复后，LookAhead 应能够在 PIR static graph 中包装 `paddle.optimizer.SGD`，创建并更新自身状态，并接受 PIR graph 中产生的 loss；同时旧 static graph 的既有行为必须保持不变。

## 验收说明

- 旧 static graph 下 LookAhead 的全局 step 更新路径保持可用。
- PIR static graph 下 LookAhead 不应依赖旧 static graph 专用的全局变量/Block 假设。
- LookAhead 与其 SGD inner optimizer 应能够接受 PIR block，并完成 accumulator/state 相关组网。
- `LookAhead.minimize()` 应接受 PIR graph 中的 loss value。
- 不得通过删除测试、弱化断言或对 PIR 路径整体绕过来满足任务。

## 技术要求

- 理解 Paddle legacy static graph 与 PIR static graph 的 Program/Block/Value 差异。
- 理解 LookAhead 的 global step、`k` 周期状态和 inner optimizer 协作关系。
- 保持 legacy static graph 的兼容行为。

## 参考资料

- https://github.com/PaddlePaddle/Paddle/issues/58067
- https://github.com/PaddlePaddle/Paddle/pull/60346
