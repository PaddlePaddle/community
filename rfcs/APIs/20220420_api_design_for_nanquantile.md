# paddle.Tensor.nanquantile 设计文档

| API名称                                                      | paddle.Tensor.nanquantile              |
| ------------------------------------------------------------ | -------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Asthestarsfalll                        |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-03-20                             |
| 版本号                                                       | V1.3                                   |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                |
| 文件名                                                       | 20200301_design_for_nanquantile.md<br> |

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充API`paddle.nanquantile`以及`paddle.Tensor.nanquantile`。paddle.nanquantile 是 [paddle.quantile](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/quantile_cn.html) 的变体，即沿给定的轴计算非nan元素的分位数。

## 2、功能目标

增加API`paddle.nanquantile`以及`paddle.Tensor.nanquantile`，实现对一个张量沿指定维度计算非nan元素q分位数的功能。

## 3、意义

飞桨支持计算非nan元素分位数。

# 二、飞桨现状

目前paddle缺少相关功能实现。

API方面，已有类似功能的API，[paddle.quantile](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/stat.py#L340)，在Paddle中是一个由多个其他API组合成的API，没有实现自己的OP，其主要实现逻辑为：

1. 如未指定维度，则通过`paddle.flatten`展平处理，若指定维度，使用`paddle.moveaxis`将指定的维度放到0处理；
2. 使用`paddle.sort`得到排序后的tensor；
3. 将`q`:[0, 1]映射到`indice`:[0, numel_of_dim-1]；并对`indice`分别做`paddle.floor`和`paddle.ceil`求得需要计算的两端元素位置；
4. `paddle.lerp`计算两端元素的加权插值，作为最终结果；
5. 根据`keepdim`参数调整至对应的shape。

实际实现时，可以基于上述API进行修改。

# 三、业内方案调研

## Pytorch

Pytorch中有API`torch.nanquantile(input, q, dim=None, keepdim=False, *, out=None) -> Tensor`，以及对应的`torch.Tensor.nanquantile(q, dim=None, keepdim=False) -> Tensor`.在pytorch中，介绍为：

```
This is a variant of torch.quantile() that “ignores” NaN values, computing the quantiles q as if NaN values in input did not exist. If all values in a reduced row are NaN then the quantiles for that reduction will be NaN. See the documentation for torch.quantile().
```

### 实现方法

在实现方法上, Pytorch是通过c++ API组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/bceb1db885cafa87fe8d037d8f22ae9649a1bba0/aten/src/ATen/native/Sorting.cpp#L145)。
由于对`NaN`处理方式的不同，其支持了`pytorch.quantile`和`pytorch.nanquantile`两个API，针对NaN部分的核心代码如下：

```c++
// Convert q in [0, 1] to ranks in [0, reduction_size)
  Tensor ranks;
  if (ignore_nan) {
    // For nanquantile, compute ranks based on number of non-nan values.
    // If all values are nan, set rank to 0 so the quantile computed is nan.
    ranks = q * (sorted.isnan().logical_not_().sum(-1, true) - 1);
    ranks.masked_fill_(ranks < 0, 0);
  } else {
    // For quantile, compute ranks based on reduction size. If there is nan
    // set rank to last index so the quantile computed will be nan.
    int64_t last_index = sorted.size(-1) - 1;
    std::vector<Tensor> tl =
        at::broadcast_tensors({q * last_index, sorted.isnan().any(-1, true)});
    ranks = at::masked_fill(tl[0], tl[1], last_index);
  }
```

对于NaN值的处理，主要逻辑如下：

- 对排序完成后的数组使用`isnan`得到一个bool矩阵，NaN对应的位置为True；
- 使用`logical_not_`取反，NaN对应的位置为False；
- 在最后一维进行求和（在之前的处理中，已经将要计算的维度变换到最后一维），计算出每个位置上有效数字的个数，减去1后可将q映射到[0, dim_wo_nan-1]，dim_wo_nan即表示除去NaN的有效数字个数，对于某个位置全是NaN的情况，则会将该位置的ranks置为0；
- 后续处理与`quantile`基本一致。

## Numpy

Numpy中有API`numpy.nanquantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=<no value>, *, interpolation=None)`

介绍为：

```
Compute the qth quantile of the data along the specified axis, while ignoring nan values. Returns the qth quantile(s) of the array elements.
```



### 实现方法

以现有numpy python API组合实现，[代码位置](https://github.com/numpy/numpy/blob/v1.22.0/numpy/lib/nanfunctions.py#L1394-L1543).
处理NaN的核心代码为：

```Python
    if arr1d.dtype == object:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        c = np.not_equal(arr1d, arr1d, dtype=bool)
    else:
        c = np.isnan(arr1d)

    s = np.nonzero(c)[0]
    if s.size == arr1d.size:
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=5)
        return arr1d[:0], True
    elif s.size == 0:
        return arr1d, overwrite_input
    else:
        if not overwrite_input:
            arr1d = arr1d.copy()
        # select non-nans at end of array
        enonan = arr1d[-s.size:][~c[-s.size:]]
        # fill nans in beginning of array with non-nans of end
        arr1d[s[:enonan.size]] = enonan

        return arr1d[:-s.size], True
```

整体逻辑与`quantile`基本相同，处理NaN的逻辑如下：

- 对输入去除NaN，仅返回有效数据，如果输入全是NaN，则返回空数据；
- 调用`quantile`的API直接对有效数据进行处理，若存在空数据，则在输入前对其进行处理，使用全NaN填充，因为`numpy.quantile`对含有NaN的数据会返回NaN，这便满足了`nanquantile`结果的正确性。

# 四、对比分析

- 使用场景与功能：在维度支持上，Pytorch只支持一维，而Numpy支持多维，这里对齐Numpy的实现逻辑，同时支持一维和多维场景。
- 代码复用：Pytorch与Numpy中的`quantile`和`nanquantile`都是对输入中的`NaN`进行不同的处理，随后使用相同的计算逻辑来获得结果，代码复用性较高，因此本方案参考其设计，最大化代码利用率。

# 五、方案设计

## 命名与参数设计

API设计为`paddle.nanquantile(x, q, axis=None, keepdim=False, name=None)`及`paddle.Tensor.nanquantile(q, axis=None, keepdim=False, name=None)`，额外添加了函数`_compute_quantile(x, q, axis=None, keepdim=Flase, ignore_nan=False)`，当`ignore_nan=True`时，表示计算`nanquantile`。

命名与参数顺序为：形参名`input`->`x`和`dim`->`axis`,  与paddle其他API保持一致性，不影响实际功能使用。
参数类型中，`axis`支持`int`与`1-D Tensor`输入，以同时支持一维和多维的场景。

## 底层OP设计

使用已有API进行修改，不再单独设计底层OP。

## API实现方案

主要按下列步骤进行组合实现, 实现位置为`paddle/tensor/stat.py`与`quantile` 、`mean`和`median`等方法放在一起：

`Pytorch`和`Numpy`中的`quantile`对含有NaN的输入将会返回NaN，而`paddle.quantile`对此不能返回正确的结果，因此对其进行修改，修改后可与`nanquantile`共用大部分代码。

1. 使用`isnan`得到标志着NaN位置的mask；

1. 对`mask`使用`paddle.logical_not`取反，在指定维度上求和，得到每个位置上的有效数字的个数，这是一个矩阵；

2. 对输入tensor使用`paddle.sort`，其中`NaN`会被排序至最后；

2. 使用第二步的**有效数字矩阵**-1乘以`q`（值域为[0, 1]）得到`indices`（值域为[-1, dim_wo_nan-1]）；

3. 针对`NaN`的处理分为了两种情况：

   1. 若为`nanquantile`：

      无需额外操作，若全是NaN，最后结果将会正确返回`NaN`；

   2. 若为`quantile`：

      使用`paddle.where`和`mask.any()`来将存在`NaN`的位置替换为对应轴上最后一个元素的索引值；

4. 对`indice`分别做`paddle.floor`和`paddle.ceil`求得需要计算的两端元素位置，对于`nanquantile`中全是`NaN`的位置，将会分别得到-1和0，对于`quantile`中存在NaN的位置，将会分别得到0和0；

5. 使用`paddle.take_along_axis`取出对应`axis`和`indice`的两端元素，若索引值为-1，将会返回0.0；

6. `paddle.lerp`计算两端元素的加权插值，作为结果，只要两输入之一为`NaN`，其输出依旧是`NaN`；

9. 根据`keepdim`参数，确定是否需要对应调整结果的shape，输出即可。

上述计算逻辑实现在`_compute_quantile(x, q, axis=None, keepdim=Flase, ignore_nan=False)`中。

# 六、测试和验收的考量

测试考虑的case如下：

- 和numpy结果的数值的一致性， `paddle.nanquantile`，`paddle.Tensor.nanquantile`和`np.nanquantile`结果是否一致；
- 参数`q`为int和1-D Tensor时输出的正确性；
- 参数`axis`为int 和1-D Tensor时输出的正确性
- `keepdim`参数的正确性；
- 未输入维度时的输出正确性；
- 错误检查：`q`值不在[0，1]时能正确抛出错误；为tensor时维度大于1时正确抛出错误；
- 错误检查：`axis`所指维度在当前Tensor中不合法时能正确抛出错误。

# 七、可行性分析及规划排期

方案主要依赖`paddle.quantile`修改而成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

由于修改了`paddle.quantile`的代码，会对其造成影响。



# 名词解释

分位数：**分位数**指的就是连续分布函数中的一个点，这个点对应概率p。若概率0<p<1，随机变量X或它的概率分布的分位数Za，是指满足条件p(X≤Za)=α的实数，对于本文的离散数据来说，`paddle.nanquantile`的结果就表示数据中有q的数据都小于这个结果，有1-q的数据都大于这个结果。

# 附件及参考资料

无