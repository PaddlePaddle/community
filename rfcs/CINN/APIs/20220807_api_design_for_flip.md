# CINN flip 设计文档

| API名称                                                      | flip                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">   | 小小夏                                                    |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-08-06                                                   |
| 版本号                                                       | V1.0                                                         |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                                      |
| 文件名                                                       | 20220807_api_design_for_flip.md<br> |


# 一、概述

## 1、相关背景

flip是众多神经网络编译器中基础的算子，该算子代表的数学函数属于injection类型。它将输入的指定维度上进行元素翻转。（Injective operator, can always injectively map output axis to a single input axis. All injective operator can still be safely fused to injective and reduction.）

## 2、功能目标

flip(tensor, dim):Reverse the order of elements in an array along the given axis.
实现flip功能，在指定维度翻转元素顺序。
可以实现对所有维度，单个维度，多个指定维度的数据转化，并返回一个新的拷贝。输入需要处理的张量以及需要翻转的维度（可以为None, int, tuple）缺省时表示翻转所有维度。
```
example

    A = [[[0, 1], 
            [2, 3]], 
        [[4, 5], 
            [6, 7]]]

    flip(A, 0)
    >>>[[[4, 5], 
            [6, 7]], 
        [[0, 1], 
            [2, 3]]]
    flip(A, 1)
    >>>[[[2, 3], 
            [0, 1]], 
        [[6, 7], 
            [4, 5]]]
    flip(A)
    >>>[[[7, 6], 
            [5, 4]], 
        [[3, 2], 
            [1, 0]]]
    flip(A, (0, 2))
    >>>[[[5, 4], 
            [7, 6]], 
        [[1, 0], 
            [3, 2]]]


```

## 3、意义

为神经网络编译器 CINN 增加基础算子flip

# 二、飞桨现状

CINN框架目前不支持此功能，暂时没有比较好的 API 替代，因此有必要实现flip算子


# 三、业内方案调研



tensorflow中调用了numpy的flip实现：

```python
@array_function_dispatch(_flip_dispatcher)
def flip(m, axis=None):
    """
    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Input array.
    axis : None or int or tuple of ints, optional
         Axis or axes along which to flip over. The default, 
         axis=None, will flip over all of the axes of the input array.
         If axis is negative it counts from the last to the first axis.
         If axis is a tuple of ints, flipping is performed on all of the axes
         specified in the tuple.
         .. versionchanged:: 1.15.0
            None and tuples of axes are supported
    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).
    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[..., ::-1, ...]`` with ``::-1`` at position n.
    flip(m) corresponds to ``m[::-1, ::-1, ..., ::-1]`` with ``::-1`` at all
    positions.
    flip(m, (0, 1)) corresponds to ``m[::-1, ::-1, ...]`` with ``::-1`` at
    position 0 and position 1.
    """
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    if axis is None:
        indexer = (np.s_[::-1], ) * m.ndim
    else:
        axis = _nx.normalize_axis_tuple(axis, m.ndim)
        indexer = [np.s_[:]] * m.ndim
        for ax in axis:
            indexer[ax] = np.s_[::-1]
        indexer = tuple(indexer)
    return m[indexer]
```

该方案特点是实际返回了一个原始array的视图，不需要新的空间，时间复杂度低。

pytorch则返回了一个新的tensor：

```
torch.flip(input, dims) → Tensor
Reverse the order of a n-D tensor along given axis in dims.

NOTE

torch.flip makes a copy of input's data. This is different from NumPy's np.flip, which returns a view in constant time. Since copying a tensor's data is more work than viewing that data, torch.flip is expected to be slower than np.flip.
}
```

# 四、对比分析

在业界，pytorch和numpy的实现基本相同，但前者产生一个数据拷贝，而后者仅仅返回一个视图。

# 五、设计思路与实现方案

可以实现对所有维度，单个维度，多个指定维度的数据转化，并返回一个新的拷贝。

## 命名与参数设计

- A：输入张量
- dim:需要翻转的维度（可以为None, int, tuple）缺省时表示翻转所有维度

## 底层OP设计

1. 在 `cinn/hlir/op/contrib/flip.h` 里声明`flip`算子。
2. 在 `cinn/hlir/op/contrib/flip.cc` 里实现`flip`算子和 `strategy`。

## API实现方案

实现目标为对于张量 A = (M, N, K)，flip( A, dim) 结果尺寸为 A = (M, N, K) 不变，但其中的数值顺序发生变化，dim维度上的数据发生翻转。

1. 在 `cinn/frontend/net_build.h` 里声明 `NetBuilder::flip`。
2. 在 `cinn/frontend/net_build.cc` 里实现 `NetBuilder::flip`。
3. 在 `cinn/pybind/frontend` 对 Python 类 `NetBuilder` 添加 `flip` 接口，并绑定到 `NetBuilder::flip`。
4. 上层 `load_paddle_model` 调用提交到 `cinn/frontend/paddle_model_to_program.h` 和 `.cc` 文件下。

通过使用 Builder 类的方法调用 flip。

```python
builder = CinnBuilder("test_basic")
a = builder.create_input(Float(32), (32, 16, 16), "A")
b = builder.flip(a, 1)
b = builder.flip(a, (0, 1))
b = builder.flip(a)
```

# 六、测试和验收的考量

1. 提供基础的 demo 文件。
2. 在`cinn/hlir/op/contrib/flip_test.cc`中添加对底层OP进行测试的代码，在`cinn/frontend/net_builder_test.cc`中添加对前端的测试。
3. 提交 API 使用方法到相应的文档中。

# 七、可行性分析和排期规划

- 可行性分析：非常可行
- 排期规划：底层OP设计已完成，API、测试和文档部分预计15天内完成

# 八、影响面

对其他模块无影响。
