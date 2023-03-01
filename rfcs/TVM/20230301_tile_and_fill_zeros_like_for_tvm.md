# 标题

| 任务名称                                                       | xxx                                              |
| -------------------------------------------------------------- | ------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">     | 梁嘉铭                                           |
| 提交时间<input type="checkbox" class="rowselector hidden">     | 2023-03-01                                       |
| 版本号                                                         | V1.0                                             |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                          |
| 文件名                                                         | 20230301_tile_and_fill_zeros_like_for_tvm.md<br> |

## 方案描述

为 tvm 支持 paddlepaddle 中的 tile 和 fill_zero_like 算子

## tile

根据参数 repeat_times 对输入 x 的各维度进行复制。平铺后，输出的第 i 个维度的值等于 x.shape[i]*repeat_times[i] 。

> 注意：其中 repeat_times 有三种情况：
> 1. repeat_times 是一个list|tuple，元素为整数 (attr.repeat_times)
> 2. repeat_times 是一个一维 Tensor，元素为整数 (input.RepeatTimes)
> 3. repeat_times 是一个list|tuple，元素为一维 Tensor (input.repeat_times_tensor)

在tvm 中，tile 已经有了实现，见[tile](https://tvm.apache.org/docs/reference/api/python/relay/index.html?highlight=tile)，但是 paddle 中的 tile 算子的参数 repeat_times 可以是一个一维 Tensor，所以需要进行判断转换。

## fill_zero_like

根据输入 x 的形状创建一个全0 Tensor，如果有 dtype 参数，则将 Tensor 的数据类型设置为 dtype，否则将 Tensor 的数据类型设置为 x 的数据类型。

其中tvm 已经有了实现full_like的实现，见[full_like](https://tvm.apache.org/docs/reference/api/python/relay/index.html?highlight=full_like)，所以只需要将输入y 的值设置为0即可。

## 单测设计

### tile

tile 主要测试：

1. 测试 paddle 中的 tile 算子的参数 repeat_times 可以是一个list|tuple，也可以是一个一维 Tensor，且长度必须和输入 x 的维度相同。

### fill_zero_like

fill_zero_like 主要测试：

1. 测试 paddle 中的 fill_zero_like 算子的参数 dtype 输入为 None 时，输出 Tensor 的数据类型为输入 Tensor 的数据类型，以及输入 dtype 时，输出 Tensor 的数据类型为 dtype。

## PR

[【PaddlePaddle Hackathon 4】add tile and fill_zeros_like for paddle frontend](https://github.com/apache/tvm/pull/14158/)