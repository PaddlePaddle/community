# paddle.vander 设计文档

| API名称                                                      | paddle.vander                               |
| ------------------------------------------------------------ | ----------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | Li-fAngyU                          |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-27                                |
| 版本号                                                       | V1.0                                      |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | Develop                                   |
| 文件名                                                       | 20230227_api_design_for_vander.md<br> |


# 一、概述

## 1、相关背景

Paddle目前没有paddle.vander API的实现。该API用于构造范德蒙矩阵。

## 2、功能目标

在飞桨中增加 paddle.vander 和 paddle.Tensor.vander API。

## 3、意义

飞桨将支持 paddle.vander 和 paddle.Tensor.vander API。

# 二、飞桨现状

飞桨中还没有 vander 的实现，但可以利用已有的 paddle.cumprod API进行实现，可参考下面的API实现方案。

# 三、业内方案调研

PyTorch：PyTorch 针对vander API有两种方案分别为 `torch.linalg.vander(x, N=None)` 和`torch.vander(x, N=None, increasing=False)`：

两者的区别是，`torch.linalg.vander` 返回的一定是升序的vander矩阵，而`torch.vander`默认返回的是降序的vander矩阵，可以通过设置`increasing=True`返回升序的vander矩阵。

因返回升序或者降序的`vander`矩阵可通过`vander = vander[:, ::-1]`快速实现，因此下面仅列出`torch.linalg.vander`方法的核心代码：

```c++
Tensor linalg_vander(
    const Tensor& x,
    c10::optional<int64_t> N) {
  auto t = x.scalar_type();
  TORCH_CHECK(t == ScalarType::Float ||
              t == ScalarType::Double ||
              t == ScalarType::ComplexFloat ||
              t == ScalarType::ComplexDouble ||
              c10::isIntegralType(t, false),
              "linalg.vander supports floating point, complex, and integer tensors, but got ", t);
  const auto x_ = x.dim() == 0 ? x.unsqueeze(-1) : x;

  auto shape = x_.sizes().vec();
  const auto n = N.value_or(shape.back());
  TORCH_CHECK(n > 1, "N must be greater than 1.");

  // Append cumprod of the oher 0...n-1 powers
  shape.push_back(n - 1);
  auto result = at::cumprod(x_.unsqueeze(-1).expand(shape), -1);
  // The row of ones
  shape.back() = 1LL;
  auto ones =  result.new_ones(shape);
  return at::cat({std::move(ones), std::move(result)}, /*dim=*/ -1);
}
```

可以看到 `torch.linalg.vander` 实现方案核心依赖于`cumprod` API，实现步骤如下：
* 对输入 `x` 检查格式。
* 检查输入 `N` 是否满足大于1。
* 通过`cumpord`构建vander矩阵的前`N-1`列，最后通过`cat`将幂为0的那一列添加到矩阵的左边。

经测试，torch.vander并不支持反向计算梯度，测试代码如下：
```python
import torch
a = torch.Tensor([1.,2.,3.])
a.requires_grad = True
b = torch.vander(a,3)
b.sum().backward()
# 报错信息：
# RuntimeError                              Traceback (most recent call last)
# /tmp/ipykernel_3602051/2253650649.py in 
#      3 a.requires_grad = True
#      4 b = torch.vander(a,3)
# ----> 5 b.sum().backward()
# RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [3, 2]], which is output 0 of SliceBackward, is at version 3; expected version 2 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True)。
```

Numpy： Numpy也有`numpy.vander(x, N=None, increasing=False)` API，其输入参数与`torch.vander`一致，核心代码如下：
```python
x = asarray(x)
if x.ndim != 1:
    raise ValueError("x must be a one-dimensional array or sequence.")
if N is None:
    N = len(x)

v = empty((len(x), N), dtype=promote_types(x.dtype, int))
tmp = v[:, ::-1] if not increasing else v

if N > 0:
    tmp[:, 0] = 1
if N > 1:
    tmp[:, 1:] = x[:, None]
    multiply.accumulate(tmp[:, 1:], out=tmp[:, 1:], axis=1)

return v
```

numpy.vander 默认返回的是降序的 vander 矩阵，可以通过设置`increasing=True`返回升序的`vander`矩阵。

其核心代码的主要流程如下:
* 将输入`x`转化为`numpy.nparray`类型，意味着numpy是能够支持`list, tuple`类型的。
* 然后检查输入`x`是否满足1维向量的条件, 并根据输入`N`是否为None对输入`N`进行赋值。
* 根据输入`x`和`N`创建维度为`(len(x), N)`的空矩阵`v`。
* 根据输入`increasing`决定是否对空矩阵`v`进行翻转。
* 利用`multiply.accumulate`累乘API对矩阵进行赋值。

Tensorflow： Tensorflow可以通过调用tensorflow.experimental.numpy.vander来构造vander矩阵，因直接采样的 numpy API，所以不再赘述。

# 四、对比分析

通过上述分析可以发现，`numpy.vander`和`torch.linalg.vander`的核心实现都是依据累乘API来实现的，且`numpy.vander`和`torch.vander`的输入参数和返回值除类型分别为`numpy.nparray`和`torch.Tensor`之外基本一致。但是`torch.vander`仅能支持输入`x`为Tensor，不像`numpy.vander`能够额外支持`list和tuple`。

经测试，当 `N` 为 0 时，`numpy.vander`以及`torch.vander`都能够正常输出维度为 `(len(x),0)` 的空 `ndarray` 或 `Tensor`。

# 五、设计思路与实现方案
经测试,`paddle.vander`可以利用已有的API组合实现，因此不需要写C++算子，且该 API 没有反向（与torch保持一致）。

## 命名与参数设计

```python
paddle.vander(x, n=None, increasing=False, name=None)
```

参数类型要求：

* 输入`x`为 1-D Tensor, 数据类型支持 int32、int64、float32、float64、complex64、complex128。
* 输入`n`的类型为int。
* 输入`increasing`的类型为bool。

参数与文档要求进行对齐。并与`numpy.vander`和`torch.vander`对齐，当 `n` 为0时，输出维度为 `(len(x),0)` 的空`Tensor`。

## API实现方案

在`python/paddle/tensor/math.py`中增加`vander`函数，并添加英文描述
```python
def vander(x, n=None, increasing=False, name=None):
    if x.dim() != 1:
        raise ValueError(
                "The input of x is expected to be a 1-D Tensor."
                "But now the dims of Input(X) is %d."
                % x.dim())
    
    if n < 0:
        raise ValueError("N must be non-negative.")
    if n is None:
        n = len(x)
    
    res = paddle.empty([len(x), n], dtype=x.dtype)

    if n > 0:
        res[:, 0] = 1
    if n > 1:
        res[:, 1:] = x[:, None]
        res[:, 1:] = paddle.cumprod(res[:, 1:], dim=-1)
    res = res[:, ::-1] if not increasing else res
    return res
```

## 单测及文档填写
在` python/paddle/fluid/tests/unittests/`中添加`test_vander_op.py`文件进行单测, 测试代码使用numpy计算结果后对比，与numpy对齐。测试代码中包含静态图/动态图下的测试。

在` docs/api/paddle/`中添加中文API文档。

# 六、测试和验收的考量

* 输入合法性及有效性检验。
* 与numpy对比结果是否一致。
* CPU、GPU测试。
* 静态图/动态图测试。
* n 为0时与numpy是否一致。

# 八、影响面

对其它模块没有影响。

# 名词解释

# 附件及参考资料
[numpy实现](https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/twodim_base.py#L546-L634)

该RFC参考了@Rayman96 所撰写的[paddle.triu_indices设计文档](https://github.com/PaddlePaddle/community/blob/master/rfcs/APIs/20220813_api_design_for_triu_indices.md)
