# paddle.Tensor.quantile 设计文档

|API名称 | paddle.Tensor.quantile | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 陈明 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-01 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | v2.2.0 | 
|文件名 | 20200301_design_for_quantile.md<br> | 

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度，支持科学计算领域API，Paddle需要扩充API`paddle.quantile`以及`paddle.Tensor.quantile`，
## 2、功能目标
增加API`paddle.quantile`以及`paddle.Tensor.quantile`，实现对一个张量沿指定维度计算q分位数的功能。

## 3、意义
飞桨支持计算分位数

# 二、飞桨现状
目前paddle缺少相关功能实现。

API方面，已有类似功能的API，[paddle.median](https://github.com/PaddlePaddle/Paddle/blob/release/2.2/python/paddle/tensor/stat.py#L251), 在Paddle中是一个由多个其他API组合成的API，没有实现自己的OP，其主要实现逻辑为：
1. 如未指定维度，则通过`paddle.flatten`展平处理
2. 通过`paddle.topk`得到原Tensor中较大的半部分（k取对应维度size / 2 + 1)。
3. 若size是奇数，则能直接取到一个元素，直接通过`paddle.slice`切分出`第size/2`个元素即可；若为偶数，通过`paddle.slice`分别切分出`第size/2-1`和`size/2`个元素，交错相加并取均值得到结果。

但在实际实现时，不能完全直接复用上述方案，理由如下：
1. `paddle.topk`未支持一次计算多个`k`值，如果仍然采用`topk`取对应`indice`的元素，在q值为多个时需要执行多次`topk`。
2. `paddle.topk`当前GPU/CPU对`NaN`值的处理未统一；
3. `paddle.slice`只支持取一次索引，仍然无法一次处理取多个索引的情况。

# 三、业内方案调研
## Pytorch
Pytorch中有API`torch.quantile(input, q, dim=None, keepdim=False, *, out=None) -> Tensor`，以及对应的`torch.Tensor.quantile(q, dim=None, keepdim=False) -> Tensor`.在pytorch中，介绍为：
```
Computes the q-th quantiles of each row of the input tensor along the dimension dim.

To compute the quantile, we map q in [0, 1] to the range of indices [0, n] to find the location of the quantile in the sorted input. If the quantile lies between two data points a < b with indices i and j in the sorted order, result is computed using linear interpolation as follows:

a + (b - a) * fraction, where fraction is the fractional part of the computed quantile index.

If q is a 1D tensor, the first dimension of the output represents the quantiles and has size equal to the size of q, the remaining dimensions are what remains from the reduction.
```

### 实现方法
在实现方法上, Pytorch是通过c++ API组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/bceb1db885cafa87fe8d037d8f22ae9649a1bba0/aten/src/ATen/native/Sorting.cpp#L145)。
其中核心代码为，根据对`NaN`处理方式的不同，同时支持了`pytorch.quantile`和`pytorch.nanquantile`两个API：
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

  // adjust ranks based on the interpolation mode
  if (interpolation == QUANTILE_INTERPOLATION_MODE::LOWER) {
    ranks.floor_();
  } else if (interpolation == QUANTILE_INTERPOLATION_MODE::HIGHER) {
    ranks.ceil_();
  } else if (interpolation == QUANTILE_INTERPOLATION_MODE::NEAREST) {
    ranks.round_();
  }

  Tensor ranks_below = ranks.toType(kLong);
  Tensor values_below = sorted.gather(-1, ranks_below);

  // Actual interpolation is only needed for the liner and midpoint modes
  if (interpolation == QUANTILE_INTERPOLATION_MODE::LINEAR ||
      interpolation == QUANTILE_INTERPOLATION_MODE::MIDPOINT) {
    // calculate weights for linear and midpoint
    Tensor weights = interpolation == QUANTILE_INTERPOLATION_MODE::MIDPOINT
        ? at::full_like(ranks, 0.5)
        : ranks - ranks_below;

    // Interpolate to compute quantiles and store in values_below
    Tensor ranks_above = ranks.ceil_().toType(kLong);
    Tensor values_above = sorted.gather(-1, ranks_above);
    values_below.lerp_(values_above, weights);
```
整体逻辑为：

- 将Tensor的对应维度调整到最后，并进行排序处理
- `broadcast_tensors`的方式，将`q [0，1]`映射到`[0, num-1]`，并检查`NaN`值。
- `masked_fill`将对应为`NaN`的部分直接赋值到最后的index（对应值也是`NaN`)
- 采用`lerp`处理`blow`和`above`的之间的插值， 包含`NaN`的结果也是`NaN`.

## Numpy 
### 实现方法
以现有numpy python API组合实现，[代码位置](https://github.com/numpy/numpy/blob/v1.21.0/numpy/lib/function_base.py#L3876-L3980).
其中核心代码为：
```Python
    ap = np.moveaxis(ap, axis, 0)
    del axis

    if np.issubdtype(indices.dtype, np.integer):
        # take the points along axis

        if np.issubdtype(a.dtype, np.inexact):
            # may contain nan, which would sort to the end
            ap.partition(concatenate((indices.ravel(), [-1])), axis=0)
            n = np.isnan(ap[-1])
        else:
            # cannot contain nan
            ap.partition(indices.ravel(), axis=0)
            n = np.array(False, dtype=bool)

        r = take(ap, indices, axis=0, out=out)

    else:
        # weight the points above and below the indices

        indices_below = not_scalar(floor(indices)).astype(intp)
        indices_above = not_scalar(indices_below + 1)
        indices_above[indices_above > Nx - 1] = Nx - 1

        if np.issubdtype(a.dtype, np.inexact):
            # may contain nan, which would sort to the end
            ap.partition(concatenate((
                indices_below.ravel(), indices_above.ravel(), [-1]
            )), axis=0)
            n = np.isnan(ap[-1])
        else:
            # cannot contain nan
            ap.partition(concatenate((
                indices_below.ravel(), indices_above.ravel()
            )), axis=0)
            n = np.array(False, dtype=bool)

        weights_shape = indices.shape + (1,) * (ap.ndim - 1)
        weights_above = not_scalar(indices - indices_below).reshape(weights_shape)

        x_below = take(ap, indices_below, axis=0)
        x_above = take(ap, indices_above, axis=0)

        r = _lerp(x_below, x_above, weights_above, out=out)

    # if any slice contained a nan, then all results on that slice are also nan
    if np.any(n):
        if r.ndim == 0 and out is None:
            # can't write to a scalar
            r = a.dtype.type(np.nan)
        else:
            r[..., n] = a.dtype.type(np.nan)

    return r
```
整体逻辑为：

- 若未指定维度，则flatten展平处理。使用`np.moveaxis`将指定的维度放到0处理；
- 将`q [0,1]`根据shape放缩到 `indice [0, nums-1]`
- 如果`indice`是整数，表示分位数是该Tensor的元素，后续直接按`indice`取元素即可；如果仍是小数，则找到其相邻位置的两个元素,后续需要用`np.lerp`插值计算得到对应元素。
- 对输入Tensor，当`indice`为整数时，直接通过`np.partition`将其按每个`indice`分为两部分(即快速排序算法中的partition部分，不完整执行排序过程以降低时间复杂度)，`indice`位置就是`q分位数`；当size为偶数时则将两端的`indice_below`和`indice_above`都做`partition`操作，并取出两端的对应结果，并利用`np.lerp`计算插值结果。
- `NaN`的处理：对存在`NaN`的情况，使用`np.isnan`确定标志位，标志位对应的位置输出值为`NaN`.
-  Numpy支持多个维度处理，以`tuple`形式作为输入。此时的分位数计算是将指定的多个维度合并后计算得到的。

# 四、对比分析
- 使用场景与功能：在维度支持上，Pytorch只支持一维，而Numpy支持多维，这里对齐Numpy的实现逻辑，同时支持一维和多维场景。
- 实现对比：由于`pytorch.gather`和`paddle.gather`实际在秩大于1时的表现不一致；在出现多个`q`值时，pytorch可直接通过处理后的`indice`进行多维索引，paddle则需要分别索引再组合到一起。因此这里不再使用`paddle.gather`索引，改使用`paddle.take_along_axis`API进行索引。


# 五、方案设计
## 命名与参数设计
API设计为`paddle.quantile(x, q, axis=None, keepdim=False, name=None)`及`paddle.Tensor.quantile(q, axis=None, keepdim=False, name=None)`
命名与参数顺序为：形参名`input`->`x`和`dim`->`axis`,  与paddle其他API保持一致性，不影响实际功能使用。
参数类型中，`axis`支持`int`与`1-D Tensor`输入,以同时支持一维和多维的场景。


## 底层OP设计
使用已有API组合实现，不再单独设计OP。

## API实现方案
主要按下列步骤进行组合实现,实现位置为`paddle/tensor/stat.py`与`mean`,`median`等方法放在一起：
1. 使用`paddle.sort`得到排序后的tensor.
2. 将`q`:[0, 1]映射到`indice`:[0, numel_of_dim-1]；并对`indice`分别做`paddle.floor`和`paddle.ceil`求得需要计算的两端元素位置；
3. 使用`paddle.take_along_axis`取出对应`axis`和`indice`的两端元素；
4. `paddle.lerp`计算两端元素的加权插值，作为结果。
5. 根据`keepdim`参数，确定是否需要对应调整结果shape。

- 对`NaN`的处理，对原tensor采用`paddle.isnan`检查`NaN`值，包含`NaN`的，在步骤4所对应位置的元素置`NaN`。
 
# 六、测试和验收的考量
测试考虑的case如下：
- 数值准确性：和numpy结果的数值的一致性, `paddle.quantile`,`paddle.Tensor.quantile`和`np.quantile`结果是否一致；
- 数值准确性：输入含`NaN`结果的正确性；
- 入参测试：参数`q`为int和1-D Tensor时输出的正确性；
- 入参测试：参数`axis`为int 和1-D Tensor时输出的正确性；
- 入参测试：`keepdim`参数的正确性；
- 入参测试：未输入维度时的输出正确性；
- 数据类型：输入Tensor`x`的`dtype`为`float32`和`float64`时的结果正确性；
- 运行设备：在CPU/GPU设备上执行时的结果正确性；
- 运行模式：动态图、静态图下执行时的结果正确性；
- 错误检查：`q`值不在[0，1]时能正确抛出错误；为tensor时维度大于1时正确抛出错误；
- 错误检查：`axis`所指维度在当前Tensor中不合法时能正确抛出错误。

# 七、可行性分析及规划排期

方案主要依赖现有paddle api组合而成，且依赖的`paddle.lerp`已于前期合入，`paddle.take_along_axis`将于近期合入。工期上可以满足在当前版本周期内开发完成。

# 八、影响面
为独立新增API，对其他模块没有影响



# 名词解释
无
# 附件及参考资料
无
