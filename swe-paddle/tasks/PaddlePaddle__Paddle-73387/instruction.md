# 修复 `gather_tree` 对 0-size Tensor 的处理

## 详细描述

当 `gather_tree(ids, parents)` 的输入 `ids` 和 `parents` 为 0-size Tensor（即 `out.numel() == 0`）时，当前实现存在以下问题：

1. **InferMeta 层**：`GatherTreeMeta` 函数会强制检查 `ids_dims == parents_dims`，但当输入为 0-size 时，这个检查可能不必要或导致错误。
2. **符号推导层**：`GatherTreeOpInferSymbolicShape` 在遍历维度时，会对 0-size 维度添加等式约束，这在 0-size 情况下是不合适的。
3. **Kernel 层**：CPU 和 GPU kernel 在分配输出内存后，直接进入计算逻辑，对 0-size 输入会访问无效内存或执行无意义的计算。

典型表现包括：

- InferMeta 阶段抛出 shape 不匹配的异常
- Kernel 执行时出现段错误或未定义行为
- 0-size Tensor 输入无法通过 `gather_tree` 算子

例如：

```python
import numpy as np
import paddle

paddle.disable_static()

# 0-size tensor 输入
ids = np.random.randint(0, high=10, size=(0, 2, 2)).astype('int64')
parents = np.random.randint(0, high=2, size=(0, 2, 2)).astype('int64')

ids_tensor = paddle.to_tensor(ids)
parents_tensor = paddle.to_tensor(parents)

out = paddle.nn.functional.gather_tree(ids_tensor, parents_tensor)
# 期望返回 shape 为 (0, 2, 2) 的空 Tensor
```

上述调用中 `ids` 和 `parents` 的 shape 为 `[0, 2, 2]`，`out.numel() == 0`。按照 API semantics，当输入为 0-size 时，该调用应正常完成并返回正确 shape 的空 Tensor。

需要在以下位置进行修改：
- InferMeta 层：当 `ids` 的 numel 为 0 时，跳过 shape 相等性检查
- 符号推导层：在遍历维度时，跳过 0-size 维度的等式约束添加
- Kernel 层：在分配输出内存后，检查 `out->numel() == 0` 并直接返回

## 验收说明

- 当输入为 0-size Tensor 时，`gather_tree` 应正常完成，返回正确 shape 的空 Tensor
- 输出的 shape 应与输入 `ids` 一致
- 非 0-size Tensor 输入下的 `gather_tree` 行为不得退化
- 梯度计算也应正常工作（0-size Tensor 的梯度也为空 Tensor）

## 技术要求

- 熟悉 C++ 和 Paddle PHI kernel 开发
- 了解 Tensor shape、0-size Tensor 和 kernel 执行路径
- 了解 `gather_tree` 算子的输入输出语义
- 了解 Paddle CPU/GPU kernel 的多 backend 实现模式
- 了解 InferMeta 和符号推导机制
- 需要从源码编译 Paddle 以验证修改
