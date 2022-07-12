# paddle.Tensor.count_nonzero 设计文档

|API名称 | paddle.count_nonzero | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | thunder95 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-07-06 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220706_api_design_for_count_nonzero.md<br> | 

# 一、概述

## 1、相关背景
为了提升飞桨API丰富度， Paddle需要支持API`paddle.count_nonzero`的功能。
## 2、功能目标
增加API`paddle.count_nonzero`，实现沿指定轴统计张量中非零元素的个数。
## 3、意义
飞桨支持`paddle.count_nonzero`的API功能。

# 二、飞桨现状
目前paddle可由其他API组合成该功能的API，不需要实现自己的OP。

API方面已有相关功能的基础API, 其主要实现逻辑为：
1. 通过`paddle.where()`将输入x中的非零元素设置成1, 也可以通过`paddle.cast()`将元素设置成bool数据类型再转成整型int64数据1。
2. 通过`paddle.sum()`按照指定轴计算非零元素的求和, 也就是非零元素的个数。

# 三、业内方案调研
## Numpy 
### 实现方法
Numpy已经有该API的实现，[代码位置](https://github.com/numpy/numpy/blob/v1.23.0/numpy/core/numeric.py#L431-L502).
其中核心代码为：
```Python
    if axis is None and not keepdims:
        return multiarray.count_nonzero(a)

    a = asanyarray(a)

    # TODO: this works around .astype(bool) not working properly (gh-9847)
    if np.issubdtype(a.dtype, np.character):
        a_bool = a != a.dtype.type()
    else:
        a_bool = a.astype(np.bool_, copy=False)

    return a_bool.sum(axis=axis, dtype=np.intp, keepdims=keepdims)

```
整体逻辑为：

- 通过`astype`将numpy数组的数据类型改为bool型变量。
- 利用`np.sum`将bool数组按照np.intp数据类型进行统计求和。

## Pytorch
Pytorch中有API`torch.count_nonzero(input, dim=None) → Tensor`。在pytorch中，介绍为：
```
Counts the number of non-zero values in the tensor input along the given dim. If no dim is specified then all non-zeros in the tensor are counted.
```

### 实现方法
在实现方法上，Pytorch的整体逻辑与Numpy一致，[代码位置](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp#L1232-L1244)。其中核心代码为：
```c++
    Tensor count_nonzero_cuda(const Tensor& self, IntArrayRef dims){
        return (self != 0).sum(dims);
    }
}
```
整体逻辑为：
- 判断张量的元素不等于0, 得到一个bool数据类型的张量。
- 对bool结果张量直接求和, 得到bool为true的元素个数, 也就是原始张量nonzero的个数。

## Tensorflow

tensorflow也提供count_nonzero接口，[代码位置](https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/ops/math_ops.py#L2449-L2514)。

```python
    if keepdims is None:
    keepdims = False
    with ops.name_scope(name, "count_nonzero", [input]):
        input = ops.convert_to_tensor(input, name="input")
        # A scalar of 'zero' is enough as `not_equal` will broadcast.
        zero = array_ops.zeros([], dtype=input.dtype)
        return cast(
            reduce_sum(
                # int64 reduction happens on GPU
                cast(gen_math_ops.not_equal(input, zero), dtypes.int64),
                axis=axis,
                keepdims=keepdims),
            dtype=dtype)
}
```
整体逻辑为：
- 判断张量的元素不等于0, 得到一个bool数据类型的张量。
- 将bool数据类型张量转为int64。
- 对int64张量求和, 也就是原始张量nonzero的个数。


# 四、对比分析
- 在维度支持上，Pytorch和Numpy都支持指向多个轴, 可以传入int或tuple。
- Numpy 还通过keepdims支持维持原始形状。

# 五、方案设计
## 命名与参数设计
API设计为`paddle.count_nonzero(x, axis=None, keepdim=False, name=None)`
命名与参数顺序为：形参名`input`->`x`和`dim`->`axis`,  与paddle其他API保持一致性，不影响实际功能使用。
参数类型中，`axis`支持`int|list|tuple`输入,以同时支持一维和多维的场景。
通过`keepdim`支持输出结果是否维持原来的形状。

## 底层OP设计
使用已有API组合实现，不再单独设计OP。

## API实现方案
主要按下列步骤进行组合实现,实现位置为`paddle/tensor/math.py`与`sum`等方法放在一起：
1. 使用`paddle.cast`将输入数据类型转为bool.
2. 使用`paddle.cast`将第一个结果再转为int64.
3. 使用`paddle.sum`将第二步结果进行求和，得到nonzero的个数，支持`axis`以及`keepdim`。

- 对`keepdim`参数的处理，对标Numpy融合到各个API当中。

# 六、测试和验收的考量
测试考虑的case如下：

- 和numpy结果的数值的一致性, `paddle.count_nonzero`,和`np.count_nonzero`结果是否一致；
- 参数`axis`为int,tuple和list时输出的正确性；
- `keepdim`参数的正确性；
- 未输入维度时的输出正确性；
- 测试在进行反向梯度计算时结果的正确性；
- 错误检查：输入`x`不是Tensor时,能否正确抛出错误；
- 错误检查：`axis`所指维度在当前Tensor中不合法时能正确抛出错误。

# 七、可行性分析及规划排期

方案主要依赖现有paddle api组合而成，且依赖的`paddle.sum`已经在 Paddle repo 的 python/paddle/tensor/math.py [目录中](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/math.py#L910)。工期上可以满足在当前版本周期内开发完成。

# 八、影响面
为独立新增API，对其他模块没有影响

# 名词解释
无
# 附件及参考资料
无

