# paddle.Tensor.bucketize 设计文档

|API名称 | paddle.bucketize | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 李芳钰 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-8 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220708_api_design_for_bucketize.md<br> | 

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度，Paddle需要扩充API`paddle.bucketize`的功能。
## 2、功能目标
增加API`paddle.bucketize`，实现根据边界返回输入值的桶索引。
## 3、意义
飞桨支持`paddle.bucketize`的API功能。

# 二、飞桨现状
目前paddle可直接由`paddle.searchsorted`API，直接实现该功能。

paddle已经实现了[paddle.searchsorted](https://github.com/PaddlePaddle/Paddle/blob/release/2.3/python/paddle/tensor/search.py#L910)API,所以只需要调用该API既可以实现该功能。

需要注意的是`paddle.bucketize`处理的sorted_sequence特殊要求为1-D Tensor。

# 三、业内方案调研
## Numpy 
### 实现方法
以现有numpy python API组合实现，[代码位置](https://github.com/numpy/numpy/blob/v1.23.0/numpy/lib/function_base.py#L5447-L5555).
其中核心代码为：
```Python
    x = _nx.asarray(x)
    bins = _nx.asarray(bins)

    # here for compatibility, searchsorted below is happy to take this
    if np.issubdtype(x.dtype, _nx.complexfloating):
        raise TypeError("x may not be complex")

    mono = _monotonicity(bins)
    if mono == 0:
        raise ValueError("bins must be monotonically increasing or decreasing")

    # this is backwards because the arguments below are swapped
    side = 'left' if right else 'right'
    if mono == -1:
        # reverse the bins, and invert the results
        return len(bins) - _nx.searchsorted(bins[::-1], x, side=side)
    else:
        return _nx.searchsorted(bins, x, side=side)
```
整体逻辑为：

- 通过`_monotonicity`判断箱子是否单调递增或者递减。
- 然后根据`mono`和参数`right`决定是否需要反转箱子。
- 最后也是通过`searchsorted`直接返回输入对应的箱子索引。

## Pytorch
Pytorch中有API`torch.bucketize(input, boundaries, *, out_int32=False, right=False, out=None) → Tensor`。在pytorch中，介绍为：
```
Returns the indices of the buckets to which each value in the input belongs, where the boundaries of the buckets are set by boundaries. Return a new tensor with the same size as input. If right is False (default), then the left boundary is closed. 
```

### 实现方法
在实现方法上，Pytorch的整体逻辑与Numpy基本一致，[代码位置](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Bucketization.cpp)。其中核心代码为：
```c++
Tensor& bucketize_out_cpu(const Tensor& self, const Tensor& boundaries, bool out_int32, bool right, Tensor& result) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries tensor must be 1 dimension, but got dim(", boundaries.dim(), ")");
  at::native::searchsorted_out_cpu(boundaries, self, out_int32, right, nullopt, nullopt, result);
  return result;
}
```
整体逻辑为：
- 检查输入参数`boundaries`。
- 然后直接利用`searchsorted_out_cpu`返回结果。

## Tensorflow
Tensorflow`tft.bucketize(
    x: common_types.ConsistentTensorType,
    num_buckets: int,
    epsilon: Optional[float] = None,
    weights: Optional[tf.Tensor] = None,
    elementwise: bool = False,
    name: Optional[str] = None
) -> common_types.ConsistentTensorType`。在Tensorflow中，介绍为：
Returns a bucketized column, with a bucket index assigned to each input.

### 实现方法
在实现方法上，Tensorflow的API参数设计于Numpy和Pytorch都不大相同，[代码位置](https://github.com/tensorflow/transform/blob/v1.9.0/tensorflow_transform/mappers.py#L1690-L1770)。这里就不具体分析其核心代码了，因为和我们想要实现的功能有很大的差距。


# 四、对比分析
- 使用场景与功能：Pytorch会比Numpy更贴和我们想要实现的功能，因为Pytorch也是仅针对1-D Tensor，而Numpy支持多维。

# 五、方案设计
## 命名与参数设计
API设计为`paddle.bucketize(x, sorted_sequence, out_int32=False, right=False, name=None)`
命名与参数顺序为：形参名`input`->`x`,  与paddle其他API保持一致性，不影响实际功能使用。
参数类型中，`x`为N-D Tensor，`sorted_sequence`为1-D Tensor。

## 底层OP设计
使用已有API组合实现，不再单独设计OP。

## API实现方案
主要按下列步骤进行实现,实现位置为`paddle/tensor/math.py`与`searchsorted`方法放在一起：
1. 使用`len(sorted_sequence)`检验参数`sorted_sequence`的维度。
2. 使用`paddle.searchsorted`得到输入的桶索引。


# 六、测试和验收的考量
测试考虑的case如下：

- 和pytorch结果的数值的一致性, `paddle.bucketize`,和`torch.bucketize`结果是否一致；
- 参数`right`为True和False时输出的正确性；
- `out_int32`为True和False时输出dtype正确性；
- 未输入`right`时的输出正确性；
- 未输入`out_int32`时的输出正确性；
- 错误检查：输入`x`不是Tensor时,能否正确抛出错误；
- 错误检查：`axis`所指维度在当前Tensor中不合法时能正确抛出错误。

# 七、可行性分析及规划排期

方案主要依赖现有paddle api组合而成，且依赖的`paddle.searchsorted`已经在 Paddle repo 的 python/paddle/tensor/search.py [目录中](https://github.com/PaddlePaddle/Paddle/blob/release/2.3/python/paddle/tensor/search.py#L910)。工期上可以满足在当前版本周期内开发完成。

# 八、影响面
为独立新增API，对其他模块没有影响

# 名词解释
无
# 附件及参考资料
无

