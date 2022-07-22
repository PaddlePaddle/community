# paddle.vsplit设计文档

|API名称 | 新增API名称 |
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | Asthestarsfalll |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-06 |
|版本号 | V1.0 |
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 |
|文件名 | 20220706_api_design_for_vsplit.md<br> |


# 一、概述
## 1、相关背景
根据 index 或者section 将输入(一个具有两个或多个维度的张量)垂直拆分为多个张量。每个拆分都是一个输入视图。
## 2、功能目标

此任务的目标是在 Paddle 框架中，新增 vsplit API，调用路径为：paddle.vsplit 和 Tensor.vsplit。

## 3、意义
为paddle新增vsplit API。

# 二、飞桨现状

飞桨目前并不支持vsplit API，但是可以使用split API进一步封装实现。

# 三、业内方案调研

## NumPy

NumPy中有API`numpy.vsplit(ary, indices_or_sections)`，介绍为：

```
Split an array into multiple sub-arrays vertically (row-wise).

Please refer to the ``split`` documentation.  ``vsplit`` is equivalent
to ``split`` with `axis=0` (default), the array is always split along the
first axis regardless of the array dimension.
```

即为将输入按照给定的sections数量在垂直轴上（行上）划分，等效于将`split`API的axis固定为0.

[核心代码](https://github.com/numpy/numpy/blob/55aacc70cf6fd627fff3642538fa5e3b12dd7111/numpy/lib/shape_base.py#L997)：

```python
def vsplit(ary, indices_or_sections):
    if _nx.ndim(ary) < 2:
        raise ValueError('vsplit only works on arrays of 2 or more dimensions')
    return split(ary, indices_or_sections, 0)
```

## PyTorch

PyTorch中由API`torch.vsplit(input, indices_or_sections)`，介绍为：

```python
Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections. Each split is a view of input.

This is equivalent to calling torch.tensor_split(input, indices_or_sections, dim=0) (the split dimension is 0), except that if indices_or_sections is an integer it must evenly divide the split dimension or a runtime error will be thrown.

This function is based on NumPy’s numpy.vsplit().
```

[核心代码](https://github.com/pytorch/pytorch/blob/736fb7d22cc948b739db2c35aeb5ad4d19aea4f4/torch/_refs/__init__.py#L2590)：

```python
def vsplit(
    a: TensorLikeType, indices_or_sections: DimsType
) -> Tuple[TensorLikeType, ...]:
    check(
        a.ndim >= 2,
        lambda: (
            "torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with "
            + str(a.ndim)
            + " dimensions!"
        ),
    )
    if isinstance(indices_or_sections, int):
        split_size = indices_or_sections
        check(
            (split_size != 0 and a.shape[0] % split_size == 0),
            lambda: (
                "torch.vsplit attempted to split along dimension 0 "
                + ", but the size of the dimension "
                + str(a.shape[0])
                + " is not divisible by the split_size "
                + str(split_size)
                + "!"
            ),
        )
        return tensor_split(a, split_size, 0)

    check(
        isinstance(indices_or_sections, (list, tuple)),
        lambda: (
            "vsplit(): received an invalid combination of arguments. "
            "Expected indices_or_sections to be of type int, list of ints or tuple of ints "
            f"but got type {type(indices_or_sections)}"
        ),
        exc_type=TypeError,
    )

    split_sizes = indices_or_sections
    return tensor_split(a, split_sizes, 0)

```

逻辑与numpy相同。

## TensorFlow

[核心代码](https://github.com/tensorflow/tensorflow/blob/e30c7b54df416a4ea0af073a580165d1d60a422e/tensorflow/python/ops/numpy_ops/np_array_ops.py#L1014-L1028)：

```python
def _split_on_axis(np_fun_name, axis):

  @np_utils.np_doc(np_fun_name)
  def f(ary, indices_or_sections):
    if isinstance(indices_or_sections, int):
      ary_shape = ary.shape[axis]
      if ary_shape is not None and ary_shape % indices_or_sections:
        raise ValueError(
            'array split does not result in an equal division')
    return split(ary, indices_or_sections, axis=axis)

  return f


vsplit = _split_on_axis('vsplit', axis=0)
```



# 四、对比分析

三种实现思路一致，只需对输入维度进行检查，使用`split`API将axis参数固定为0即可。值得注意的是，`paddle.split`与上述API的默认逻辑不同，当`indices_or_sections`与`num_or_sections`都是`整数`时，二者意义相同，表示需要分块的个数；当二者为`列表或元组`时，`indices_or_sections`表示各个分块之间的分隔索引，而`num_or_sections`则表示划分的每一个Tensort在指定axis上的大小。

# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.vsplit(x, num_or_sections, name=None)`和`Tensor.(num_or_sections, name=Nome)`。

## 底层OP设计

仅使用python实现，无需设计底层OP。

## API实现方案

按照`paddle.split`的实现逻辑，API初步实现如下，实现于`split`同文件中：

```python
def vsplit(x, num_or_sections, name=None):
    """
    Split the input tensor into multiple sub-Tensors along the vertical axis, which is equivalent to ``paddle.split`` with ``axis=0``.
    
    Args:
        x (Tensor): A Tensor whose dimension must be greater than 2. The data type is bool, float16, float32, float64, int32 or int64.
        num_or_sections (int|list|tuple): If ``num_or_sections`` is an int, then ``num_or_sections`` 
            indicates the number of equal sized sub-Tensors that the ``x`` will be divided into.
            If ``num_or_sections`` is a list or tuple, the length of it indicates the number of
            sub-Tensors and the elements in it indicate the sizes of sub-Tensors'  dimension orderly.
            The length of the list must not  be larger than the ``x`` 's size of axis 0.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        list(Tensor): The list of segmented Tensors.
    
    Example:
        .. code-block:: python
            
            import paddle
            
            # x is a Tensor of shape [8, 6, 7]
            x = paddle.rand([8, 6, 7])

            out0, out1, out2 = paddle.vsplit(x, num_or_sections=2)
            print(out0.shape)  # [4, 6, 7]
            print(out1.shape)  # [4, 6, 7]

            out0, out1, out2 = paddle.vsplit(x, num_or_sections=[1, 3, 4])
            print(out0.shape)  # [1, 6, 7]
            print(out1.shape)  # [3, 6, 7]
            print(out2.shape)  # [4, 6, 7]

            out0, out1, out2 = paddle.split(x, num_or_sections=[2, 3, -1])
            print(out0.shape)  # [2, 6, 7]
            print(out1.shape)  # [3, 6, 7]
            print(out2.shape)  # [3, 6, 7]

    """

    if x.ndim < 2:
        raise ValueError("The input tensor's dimension must be greater than 2, but got {}".format(x.ndim))

    return split(x, num_or_sections, axis=0, name=name)
```



# 六、测试和验收的考量

1. 动态图、静态图下保证结果的维度、数值正确。
2. CPU、GPU设备上保证结果的维度、数值正确。
3. 各个数据类型下保证结果的维度、数值正确。
4. 覆盖API所有参数情况、进行参数有效性和边界值测试。
5. `num_or_sections`为整数、列表或元组时保证结果的维度、数值正确。
6. 当输入维度小于2时正确抛出错误。

# 七、可行性分析和排期规划

实现较简单，可行。

排期规划为：

1. 完成API、英文文档编写。
2. 完成单测编写。
3. 完成中文文档编写。

# 八、影响面
为独立新增API，对其他模块无影响。

# 名词解释

无

# 附件及参考资料

无
