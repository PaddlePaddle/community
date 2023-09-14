# paddle.hypot设计文档

|API名称 | paddle.hypot                     | 
|---|----------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | llyyxx0413                       | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-09-14                       | 
|版本号 | V1.0                             | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                          | 
|文件名 | 20230914_api_design_for_hypot.md | 


# 一、概述
## 1、相关背景

`hypot` 函数实现直角三角形斜边长度求解的计算：

$$ out= \sqrt{x^2 + y^2} $$

## 2、功能目标

为 Paddle 新增 `paddle.hypot` & `paddle.hypot_` 和 `Tensor.hypot` & `Tensor.hypot_`API，实现直角三角形斜边长度求解的计算。

## 3、意义

为 Paddle 新增 `paddle.hypot` & `paddle.hypot_` 和 `Tensor.hypot` & `Tensor.hypot_`API，实现直角三角形斜边长度求解的计算。

# 二、飞桨现状

对飞桨框架目前不支持此功能，可用其他API组合实现的此功能，代码如下；

```Python
import paddle
import numpy as np

a = paddle.randn([3, 4])
b = paddle.randn([3, 4])

out = (a.pow(2) + b.pow(2)).sqrt()

print(out)
```

# 三、业内方案调研

## 1. Numpy
在Numpy中使用的API格式如下：
`numpy.hypot(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])` 
给定直角三角形的直角边，返回其斜边
其中:
`x1`, `x2`(array_like)：直角三角形的直角边, 如果`x1.shape != x2.shape` 则`x1`与 `x2`的`shape`必须可以广播
`out`(ndarray): 接收结果的ndarray,如果为`None`, 则返回新的ndarray
`where`, `casting`, `order`, `dtype`与Numpy 其他API保持一致。

实现代码如下:
```c++ 
NPY_INPLACE double npy_hypot(double x, double y)
{
#ifndef NPY_BLOCK_HYPOT
    return hypot(x, y);
#else
    double yx;

    if (npy_isinf(x) || npy_isinf(y)) {
        return NPY_INFINITY;
    }

    if (npy_isnan(x) || npy_isnan(y)) {
        return NPY_NAN;
    }

    x = npy_fabs(x);
    y = npy_fabs(y);
    if (x < y) {
        double temp = x;
        x = y;
        y = temp;
    }
    if (x == 0.) {
        return 0.;
    }
    else {
        yx = y/x;
        return x*npy_sqrt(1.+yx*yx);
    }
#endif
}
```


## 3. Pytorch

在 Pytorch 中使用的 API 格式如下：

`torch.hypot(input, other, *, out=None)`

其中，`input` 和 `other` 为 `Tensor` 类型，是直角三角形的边。

实现代码如下：

```c++
// cpu
void hypot_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "hypot_cpu", [&]() {
    cpu_kernel_vec(
        iter,
        [=](scalar_t a, scalar_t b) -> scalar_t {
            return std::hypot(a, b);
        },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
            return a.hypot(b);
        });
  });
}

Vectorized<T> hypot(const Vectorized<T> &b) const {
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = std::hypot(values[i], b[i]);
    }
    return ret;
  }

//gpu
void hypot_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.common_dtype(), "hypot_cuda",
      [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
            iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return ::hypot(a, b);
        });
      });
}

```

```python

//inpalce 

hypot_ = _make_inplace(hypot)
def _make_inplace(fn):
    """
    Given a function with out variant (i.e. using `out_wrapper()), it returns its in-place variant
    See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-do-in-place-operations-work-in-pytorch
    """

    # nb. We use the name of the first argument used in the unary references
    @wraps(fn)
    def _fn(a, *args, **kwargs):
        return fn(a, *args, out=a, **kwargs)

    inplace_name = f"{fn.__name__}_"
    _fn.__name__ = inplace_name
    _fn = register_decomposition(getattr(aten, inplace_name))(_fn)

    # We access the __all__ attribute of the module where fn is defined
    # There may be a cleaner way of doing this...
    from inspect import getmodule

    _all = getmodule(fn).__all__  # type: ignore[union-attr]
    if inplace_name not in _all:
        _all.append(inplace_name)
    return _fn
```


# 四、对比分析

## 1. 不同框架API使用方式

### 1. Numpy

```Python
import numpy as np

np.hypot(3*np.ones((3, 3)), 4*np.ones((3, 3)))
>>>array([[ 5.,  5.,  5.],
       [ 5.,  5.,  5.],
       [ 5.,  5.,  5.]])
```

### 2. PyTorch

```Python
import torch

a = torch.hypot(torch.tensor([4.0]), torch.tensor([3.0, 4.0, 5.0]))
>>>tensor([5.0000, 5.6569, 6.4031])
```


上述框架从使用体验来说，差异不大，都是直接调用 API 即可。实现上`Numpy`倾向于公式实现，而`torch` 倾向于使用库。出于paddle 目前新增API现状，故采用组合的方式 为 Paddle 新增 `paddle.hypot` API。

# 五、设计思路与实现方案

## 命名与参数设计

<!-- 参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) -->

API设计为 `paddle.hypot(x, y, name)`。其中，`x`, `y` 为 `Tensor` 类型，是直角三角形的边，``paddle.hypot_(x)` 为 inplace 版本。`Tensor.hypot(p)` 为 Tensor 的方法版本。`Tensor.hypot_(x)` 为 Tensor 的 方法 inplace 版本。

## API实现方案

 采用现有 PYTHON API 组合实现，实现位置为 Paddle repo `python/paddle/tensor/math.py` 目录。并在 python/paddle/tensor/init.py 中，添加 `hypot` & `hypot_` API，以支持 `paddle.Tensor.hypot` & `paddle.Tensor.hypot_` 的调用方式

# 六、测试和验收的考量

<!-- 参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) -->

可考虑一下场景：
1. 常规场景：`x`, `y`为常规输入，比较其与numpy的输出精度差异
2. Broadcast场景: `x`, `y`形状可广播，比较输出shape是否为广播后输出以及形状；
3. 0维场景，`x`, `y`为0维Tensor, 比较输出是否亦为0维Tensor  

# 七、可行性分析和排期规划

本 API 主要使用组合实现，难度不高，工期上能满足要求。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释

# 附件及参考资料