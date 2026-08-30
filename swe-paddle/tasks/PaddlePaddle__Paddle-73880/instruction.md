# 修复 `paddle.nn.functional.softmax_with_cross_entropy` 对 0-size Tensor 的处理

## 详细描述

当 `paddle.nn.functional.softmax_with_cross_entropy` 的输入 logits 中存在大小为 `0` 的 dimension（即 `softmax->numel() == 0`）时，当前 CPU/GPU/XPU kernel 实现会直接进入后续计算逻辑，导致 kernel 内部对空数据执行计算时出错或产生未定义行为。

典型表现包括：

- kernel 在处理 0-size tensor 时崩溃或报错
- 调用失败并抛出与 shape 或内存访问相关的错误

例如：

```python
import numpy as np
import paddle

paddle.disable_static()
# shape [0, 10], soft_label=False
logits = paddle.to_tensor(np.random.uniform(0.1, 1.0, (0, 10)).astype('float32'))
label = paddle.to_tensor(np.random.randint(0, 10, (0, 1), dtype='int64'))
loss, softmax = paddle.nn.functional.softmax_with_cross_entropy(logits, label, return_softmax=True)
```

上述调用中 `logits` 的 shape 为 `[0, 10]`，不包含任何元素。当 `soft_label` 为 False 时，axis 所在列不能为 0，其他列相同，因此 softmax 和 loss 的 numel 都为 0。按照 API semantics，该调用应正常完成并返回正确 shape 的空 tensor。

此外，反向传播也需要正确处理 0-size tensor 的情况，确保梯度被正确填充。

## 验收说明

- 当输入 logits 的 numel 为 0 时，`softmax_with_cross_entropy` 的前向和反向计算应正常完成
- 返回的 softmax 和 loss tensor 应保持正确 shape 和 dtype
- 非 0-size tensor 输入下的 cross entropy 行为不得退化
- CPU、GPU、XPU 后端均应正确处理 0-size 输入

## 技术要求

- 熟悉 C++/CUDA 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 cross entropy 算子的前向和反向计算语义
- 了解 Paddle CPU/GPU/XPU kernel 的实现模式
- 需要从源码编译 Paddle 以验证修改
