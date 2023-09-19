# paddle.scatter 设计文档

| API名称      | paddle.scatter                     |
| ------------ | -------------------------------------- |
| 提交作者     | mhy                                    |
| 提交时间     | 2023-09-16                             |
| 版本号       | V1.0                                   |
| 依赖飞桨版本  |  develop                                |
| 文件名       | 20230916_api_design_for_scatter.md |

# 一、概述

## 1、相关背景

`scatter` 是一个常用的API， API提供了根据index信息更新原Tensor的功能。

## 2、功能目标

当前 `paddle.scatter` API提供了根据index信息更新原Tensor的功能，但指定维度和归约方式功能尚不支持。

目前 `paddle.scatter` 的规约支持 None和sum。

## 3、意义

该API是一个常用的API，可以方便用户使用。让用户不用自己实现该功能，提高用户的使用效率。

# 二、飞桨现状

当前 `paddle.scatter` API但缺少指定轴和归约方式等功能。

paddle是通过 op kernel 的形式实现 `scatter` 和 `scatter_` API。
```python
_C_ops.scatter(x, index, updates, overwrite)
_C_ops.scatter_(x, index, updates, overwrite)

helper.append_op(
          type="scatter",
          inputs={"X": x, "Ids": index, "Updates": updates},
          attrs={'overwrite': overwrite},
          outputs={"Out": out},
      )
```

paddle 实现的 index 功能和 torch的`scatter`中的index的功能不一致。paddle 的 index 只能是一维或者0维的。

```python
        import paddle
        #input:
        x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
        # shape of updates should be the same as x
        # shape of updates with dim > 1 should be the same as input
        updates = paddle.to_tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')
        overwrite = False
        # calculation:
        if not overwrite:
            for i in range(len(index)):
                x[index[i]] = paddle.zeros([2])

        for i in range(len(index)):
            if (overwrite):
                x[index[i]] = updates[i]
            else:
                x[index[i]] += updates[i]
        # output:
        out = paddle.to_tensor([[3, 3], [6, 6], [1, 1]])
        out.shape # [3, 2]
```

# 三、业内方案调研

## Pytorch

### Pytorch中 有 API `Tensor.scatter_(dim, index, src, reduce=None) → Tensor`

在pytorch中，介绍为：

```
Writes all values from the tensor `src` into `self` at the indices specified in the `index` tensor. For each value in `src`, its output index is specified by its index in `src` for` dimension != dim` and by the corresponding value in `index` for `dimension = dim`.
```

其中输入参数的描述如下：

- dim (int) – the axis along which to index
- index (LongTensor) – the indices of elements to scatter, can be either empty or of the same dimensionality as src. When empty, the operation returns self unchanged.
- src (Tensor or float) – the source element(s) to scatter.
- reduce (str, optional) – reduction operation to apply, can be either 'sum' or 'multiply'.

`torch.scatter` 和 paddle.scatter  的区别在于:
1. torch 支持 dim 配置.

```python
# For a 3-D tensor, self is updated as:
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

2. reduce 额外支持 ‘multiply'
3. index的维度和src是一致的。


### Pytorch中 有 API `Tensor.index_reduce_(dim, index, source, reduce, *, include_self=True) → Tensor`

其中输入参数的描述如下：

- dim (int) – dimension along which to index
- index (Tensor) – indices of source to select from, should have dtype either torch.int64 or torch.int32
- source (FloatTensor) – the tensor containing values to accumulate
- reduce (str) – the reduction operation to apply ("prod", "mean", "amax", "amin")

`torch.index_reduce_` 和 paddle.scatter 的区别在于:
1. torch 支持 dim 配置.

```python
self[index[i], :, :] *= src[i, :, :]  # if dim == 0
self[:, index[i], :] *= src[:, i, :]  # if dim == 1
self[:, :, index[i]] *= src[:, :, i]  # if dim == 2
```

2. reduce 支持 mean, amax, amin, 但不支持 add
3. index的维度是一维的。
4. paddle 只支持 include_self = False。


## Tensorflow

Tensorflow 没有提供 `scatter` 的API。

# 四、对比分析
- Pytorch 自定义Kernel的方式更加高效. index支持多维度，支持指定dim和reduce方式。
- Tensorflow 不支持 scatter API

# 五、方案设计

## 命名与参数设计

为了兼容现有API，并支持axis, reduce的功能扩展，scatter API 按如下接口设计：

```python
# https://github.com/PaddlePaddle/Paddle/blob/release/2.5/python/paddle/tensor/manipulation.py#L2849
paddle.scatter(x, index, updates, overwrite=True,  axis=0, reduce='sum', name=None)
paddle.scatter_(x, index, updates, overwrite=True, axis=0, reduce='sum', name=None)

scatter 参数如下：
```
- `x (Tensor)` - ndim > = 1 的输入 N-D Tensor。数据类型可以是 float32，float64。
- `index （Tensor）`- 一维或者零维 Tensor。数据类型可以是 int32，int64。 index 的长度不能超过 updates 的长度，并且 index 中的值不能超过输入的长度。
- `updates （Tensor` - 根据 index 使用 update 参数更新输入 x。当 index 为一维 tensor 时，updates 形状应与输入 x 相同，并且 dim>1 的 dim 值应与输入 x 相同。当 index 为零维 tensor 时，updates 应该是一个 (N-1)-D 的 Tensor，并且 updates 的第 i 个维度应该与 x 的 i+1 个维度相同。
- `overwrite （bool，可选)`- 指定索引 index 相同时，更新输出的方式。如果为 True，则使用覆盖模式更新相同索引的输出，如果为 False，则根据`reduce`参数指定的模式更新相同索引的输出。默认值为 True。
- `axis (int, 可选)` - 要索引的维度。默认值为0.
- `reduce(str,可选)` - 指定规约运算，可以是 sum、mul, mean, amax, amin。默认值为 sum.
-  `include_self (bool，可选)` - arr 张量中的元素是否包含在规约中。默认值 include_self = False.
- `name (str，可选)` - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。


相比于 torch.scatter 和 torch.index_reduce ：
1. 新增 axis 属性，支持按axis所以，实现方式同 torch.index_reduce。
2. 新增 reduce 属性，支持 sum、mul、mean, max, min 规约方式，实现方式同  torch.scatter、torch.index_reduce .
3. 相比于 torch.index_reduce 中的 include_self=True，paddle.scatter 保持不变，仍为 include_self=False 。
4. assign 规约，任由 overwrite 控制，reduce 中不新增。

## 底层OP设计

增强现有 scatter op 实现，支持指定dim和指定reduce方法。具体为`_C_ops.scatter`和`_C_ops.scatter_`. OP 的参数遵循API的参数与顺序。

axis 使用如下规则索引：

```python
self[index[i], :, :] *= src[i, :, :]  # if axis == 0
self[:, index[i], :] *= src[:, i, :]  # if axis == 1
self[:, :, index[i]] *= src[:, :, i]  # if axis == 2
```

为了保证兼容性，保留 `overwrite` 字段。只有到 `overwrite` 为False时，按照 `reduce` 选择规约方式；否则则按照 `assign` 逻辑处理。


反向梯度计算逻辑如下：

```c++
  if (reduce == "prod") {
    Tensor masked_self = self.masked_fill(self == 0, 1);
    Tensor masked_self_result = masked_self.index_reduce(dim, index, source, reduce, include_self);
    grad_self = grad * masked_self_result / masked_self;
    Tensor src_zero = source == 0;
    Tensor src_num_zeros = zeros_like(self).index_add(dim, index, src_zero.to(self.dtype())).index_select(dim, index);
    Tensor src_single_zero = bitwise_and(src_zero, src_num_zeros == 1);
    // For src positions with src_single_zero, (grad * result).index_select(dim,index) / source.masked_fill(src_zero, 1)
    // would incorrectly propagate zeros as the gradient
    Tensor masked_src = source.masked_fill(src_single_zero, 1);
    Tensor masked_src_result = self.index_reduce(dim, index, masked_src, reduce, include_self);
    Tensor grad_src1 = where(src_single_zero,
                             (grad * masked_src_result).index_select(dim, index),
                             (grad * result).index_select(dim, index) / source.masked_fill(src_zero, 1));
    if ((src_num_zeros > 1).any().item<bool>()) {
      auto node = std::make_shared<DelayedError>(
        "index_reduce(): Double backward is unsupported for source when >1 zeros in source are scattered to the same position in self",
        /* num inputs */ 1);
      auto result = node->apply({ grad_src1 });
      grad_src = result[0];
    } else {
      grad_src = grad_src1;
    }
  } else if (reduce == "mean") {
    Tensor N = include_self ? ones_like(grad) : zeros_like(grad);
    N = N.index_add(dim, index, ones_like(source));
    N.masked_fill_(N == 0, 1);
    grad_self = grad / N;
    Tensor N_src = N.index_select(dim, index);
    grad_src = grad.index_select(dim, index) / N_src;
  } else if (reduce == "amax" || reduce == "amin") {
    Tensor value = result.index_select(dim, index);
    Tensor self_is_result = (self == result).to(self.scalar_type());
    Tensor source_is_result = (source == value).to(self.scalar_type());
    Tensor N_to_distribute = self_is_result.index_add(dim, index, source_is_result);
    Tensor grad_distributed = grad / N_to_distribute;
    grad_self = self_is_result * grad_distributed;
    grad_src = source_is_result * grad_distributed.index_select(dim, index);
  } else {
    AT_ERROR("Expected 'reduce' to be one of 'prod', 'amax', 'amin' or 'mean' but got ", reduce, ".");
  }

  if (!include_self) {
    grad_self = grad_self.index_fill(dim, index, 0);
  }
```

reduce=sum 的计算逻辑和 mean 类似。

## API实现方案

[API已有](https://github.com/PaddlePaddle/Paddle/blob/release/2.5/python/paddle/tensor/manipulation.py#L2849)，需要新增axis和reduce参数。

## 代码实现文件路径

函数API实现路径: python/paddle/tensor/manipulation.py

单元测试路径：在 Paddle repo 的 test/ 目录, 同时在 paddle/test/legacy_test/test_inplace.py、paddle/test/legacy_test/test_scatter_op.py 修改对应的单侧。


# 六、测试和验收的考量

测试考虑的case如下：

- 不同 dim 下功能是否符合预期。
- 不同 reduce 下功能是否符合预期。
- 验证反向梯度是否符合预期。

# 七、可行性分析及规划排期

方案实施难度可控，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
