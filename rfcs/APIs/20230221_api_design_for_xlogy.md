# xlogy 设计文档

| API名称                                                      | 新增API名称                               |
| ------------------------------------------------------------ | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | paddle.xlogy                         |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-21                                |
| 版本号                                                       | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                   |
| 文件名                                                       | 20230221_api_design_for_xlogy.md<br> |


# 一、概述

## 1、相关背景

paddle.xlogy API 用于实现分段函数计算 input * log(other), 当 other 为 NaN时，输出为 NaN，当乘子 input 为 0 时输出为 0.

## 2、功能目标

在飞桨中增加 paddle.xlogy API。

## 3、意义

飞桨将支持 paddle.xlogy API。

# 二、飞桨现状

飞桨中还没有 xlogy，直接使用 input * log(other), 无法达到分段函数的需求，需要额外增加对 float('inf') 和 float('nan') 等情况的处理。


# 三、业内方案调研

PyTorch：PyTorch 支持 xlogy，CPU Kernel 实现如下：

```c++
void xlogy_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "xlogy_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t x, scalar_t y) -> scalar_t {
      if (at::_isnan(y)){
        return NAN;
      }
      if (x == 0){
        return 0;
      }
      return x * std::log(y);
    });
  });
}
```

可以看到就是利用 if 语句判断 y 是否为 NAN，以及 x 是否等于 0。需要注意的时判断 y 是否为 NAN 的优先级大于判断 x 是否为0。CUDA Kernel 和 CPU Kernel 是相似的。

Scipy：Scipy 支持 xlogy，实现如下：

```python
cdef inline number_t xlogy(number_t x, number_t y) nogil:
    if x == 0 and not zisnan(y):
        return 0
    else:
        return x * zlog(y)
```

可以看到也是直接利用 if 语句进行分段的实现思路。但是与 torch 不同，scipy 没有对 NAN 值单独进行判断和return的操作。这是可以理解的因为 log(NAN) 也是NAN， NAN 与任何数进行运算还是 NAN。

# 四、对比分析

PyTorch 和 Scipy 的思路大体是类似的，但是要求的 Paddle 实现是 python 实现而不是在 kernel 层面上进行实现，所以需要进行一定的转换。

# 五、设计思路与实现方案

## 命名与参数设计

```python
paddle.xlogy(x, other, name=None)
```

参数与文档要求进行对齐。

## API实现方案

参考 Scipy 的处理方式，仅处理输入 x == 0 且 other != Nan 时的情况。并利用 log(1.) = 0 的特点完成该API的实现，且前向输出和反向梯度于torch进行对齐。

注： x == 0 且 other != Nan （可以等价替换为） x == 0 且 log(other) != Nan。

因此有如下等价替换过程： 
`x == 0 且 other != Nan` <==> `x == 0 且 log(other) != Nan` <==> `x == 0 且 (other <=0 或 other == inf）`

Paddle：xlogy API 代码：
```python
def xlogy(x, other, name=None):
    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'xlogy')
    mask = (x == 0) & ((other <= 0) | (other == float('inf')))
    other = paddle.where(mask, paddle.ones(other.shape, other.dtype), other)
    return x * paddle.log(other)
```

1.首先对变量的dtype进行检查。

2.然后利用`mask = (x == 0) & ((other <= 0) | (other == float('inf')))`提取满足`x == 0 且 (other <=0 或 other == inf）`条件的位置信息。

3.利用`paddle.where`将满足上述条件的输入`other`替换成 1.0 ,使得`log(other)`刚好为 0.0。

4.最后直接返回`x * paddle.log(other)`。


# 六、测试和验收的考量

因numpy没有xlogy的实现，所以利用scipy.special.xlogy进行前向测试和验收：

1. 测试 API 在动态图和静态图下与 scipy 的一致性。
2. 测试 CPU、GPU 上与 scipy.special.xlogy 的一致性。
3. 测试反向梯度的正确性。
4. 测试输入`x`和`other`各自包含 `NAN,0,inf,-inf` 等情况下与 scipy的一致性。

# 八、影响面

对其它模块没有影响。

# 名词解释

# 附件及参考资料
[scipy实现](https://github.com/scipy/scipy/blob/main/scipy/special/_xlogy.pxd)

[torch实现](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/BinaryOpsKernel.cpp#L1081)