# 对齐 Tensor 与 Layer 的转换调用行为

Paddle 的 Tensor 和神经网络 Layer 转换接口目前无法一致地支持常见调用形式。请调整它们的可观察行为，使下列用法能够可靠工作。

## 行为要求

- 接受 dtype、device 或参考 Tensor 作为第一个位置参数。
- 在适用场景中，支持同时以位置参数传入 device 和 dtype。
- 支持以关键字参数传入 `device` 和 `dtype`。
- 支持 `blocking`、`non_blocking` 和 `copy` 的合法位置参数或关键字参数形式。
- 允许不传参数的转换调用。
- 同时指定 `blocking` 和 `non_blocking` 时抛出 `TypeError`。
- 位置参数过多或存在未知关键字参数时抛出 `TypeError`。
- 第一个位置参数无法识别时抛出 `ValueError`。
- Layer 转换应原地执行、返回同一个 Layer 对象，并递归应用于子层。
- 传入参考 Tensor 时，目标对象应采用其转换属性。

## 验收标准

- Tensor 转换支持 dtype、device、参考 Tensor、阻塞选项和复制语义。
- Layer 转换支持对应的 dtype、device、参考 Tensor 和阻塞选项调用，并保持对象身份不变。
- 嵌套 Layer 和已注册 buffer 均得到一致转换；按照本任务目标行为，dtype 转换也适用于整数 buffer。
- 现有 Layer 转换回归测试继续通过。
