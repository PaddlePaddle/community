# 标题

| 任务名称                                                       | xxx                                       |
| -------------------------------------------------------------- | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 梁嘉铭                                    |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2023-03-01                                |
| 版本号                                                         | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                   |
| 文件名                                                         | 20230301_mish_and _unstack_for_tvm.md<br> |

## 方案描述

为 tvm 支持 paddlepaddle 中的 mish 和 unstack 算子。

## mish

mish 是一种激活函数，其公式为：

$$
\text{mish}(x) = x * \tanh(\text{softplus}(x)) = x * \tanh(\ln(1 + \exp(x)))
$$

在tvm 中无法直接支持 mish，需要将其转换为其他算子的组合，如下图所示：

```python
def mish(x):
    exp = tvm.te.exp(x)
    softplus = tvm.te.log(exp + tvm.te.const(1.0, exp.dtype))
    tanh = tvm.te.tanh(softplus)
    return x * tanh
```

## unstack

unstack 是将一个 tensor 按照指定的 axis 拆分为多个 tensor 的算子

其参数为：

| 参数名 | 类型        | 说明                     |
| ------ | ----------- | ------------------------ |
| x      | Tensor      | 输入 tensor              |
| axis   | int         | 指定的 axis              |
| num    | int（可选） | 指定拆分后的 tensor 数量 |

其输出为：

| 参数名 | 类型   | 说明            |
| ------ | ------ | --------------- |
| y      | Tensor | 拆分后的 tensor |

在 tvm 中无法直接支持 unstack，需要将其转换为其他算子的组合，如下图所示：

```python
def unstack(x, axis, num):
    shape = x.shape
    if axis < 0:
        axis = axis + len(shape)
    if num is None:
        num = shape[axis]
    out = []
    for i in range(num):
        begin = [0] * len(shape)
        end = shape
        begin[axis] = i
        end[axis] = i + 1
        out.append(tvm.te.strided_slice(x, begin, end, [1] * len(shape)))
    return out
```

> 注：unstack 的实现中，如果 num 为 None，则需要根据 axis 的值，从 shape 中获取 num 的值。
> 
> 若 axis < 0 则 axis = axis + len(shape)


## PR

[【PaddlePaddle Hackathon 4】add unstack and mish op for paddle frontend](https://github.com/apache/tvm/pull/14159)