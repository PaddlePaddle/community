# paddle.polygamma 设计文档

|API名称 | paddle.polygamma | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 吃点儿好的 [paddle](https://github.com/KateJing1212/community) | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-03-22 | 
|版本号 | V2.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20200322_api_design_for_polygamma.md | 


# 一、概述
## 1、相关背景

对于输入张量，其 digamma 函数的 n 阶导，称为多伽马函数`polygamma`。

## 2、功能目标

为 Paddle 新增 polygamma API，用于对输入张量的digamma函数进行n阶导操作。

## 3、意义

为 Paddle 新增 polygamma API，用于对输入张量的digamma函数进行n阶导操作。

# 二、飞桨现状

对飞桨框架目前不支持此功能，但支持 digamma 函数[(参考API文档)](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/digamma_cn.html) ，且支持输入为张量。digamma 函数是 gamma 函数的对数的一阶导数，k 阶的 polygamma 函数是 gamma 函数的对数的 (k + 1) 阶导数。

```Python
def digamma(x, name=None):
    r"""
    Calculates the digamma of the given input tensor, element-wise.
    .. math::
        Out = \Psi(x) = \frac{ \Gamma^{'}(x) }{ \Gamma(x) }
    Args:
        x (Tensor): Input Tensor. Must be one of the following types: float32, float64.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the digamma of the input Tensor, the shape and data type is the same with input.
    Examples:
        .. code-block:: python
            import paddle
            data = paddle.to_tensor([[1, 1.5], [0, -2.2]], dtype='float32')
            res = paddle.digamma(data)
            print(res)
            # Tensor(shape=[2, 2], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #       [[-0.57721591,  0.03648996],
            #        [ nan       ,  5.32286835]])
    """

    if in_dygraph_mode():
        return _C_ops.digamma(x)
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.digamma(x)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], 'digamma')
    helper = LayerHelper('digamma', **locals())
    out = helper.create_variable_for_type_inference(x.dtype)
    helper.append_op(type='digamma', inputs={'X': x}, outputs={'Out': out})
    return out
```

# 三、业内方案调研

## 1. pytorch

在 pytorch 中使用的 API 格式如下：

`torch.polygamma(input, n, *, out=None) -> Tensor`[(参考API文档)](https://pytorch.org/docs/stable/generated/torch.polygamma.html)

上述函数参数中，`input` 是一个Tensor，表示要计算polygamma函数的输入值；`n`是一个整数，表示要计算的polygamma函数的阶数；`out`是一个可选的输出Tensor，用于存储计算结果。

实现的代码如下：

```cpp
Tensor polygamma_backward(const Tensor& grad_output, const Tensor& self, int64_t n) {
  checkBackend("polygamma_backward", {grad_output, self}, Backend::CPU);
  auto input = self;
  auto mask = input >= 0;
  input = input.abs();
  auto grad_input = at::empty_like(input);
  if (input.numel() > 0) {
    grad_input.masked_scatter_(mask, at::digamma(input.masked_select(mask)) + at::polygamma(n + 1, input.masked_select(mask)));
    grad_input.masked_scatter_(~mask, at::nan(at::dtype(at::kFloat)));
  }
  return grad_output * grad_input;
}

Tensor polygamma(const Tensor& self, int64_t n) {
  checkBackend("polygamma", {self}, Backend::CPU);
  auto result = at::empty_like(self);
  AT_DISPATCH_FLOATING_TYPES(self.type(), "polygamma_cpu", [&] {
    polygamma_stub<scalar_t>::apply(result, self, n);
  });
  return result;
}

```
可以看到，`polygamma`函数调用了`polygamma_stub`函数来计算`polygamma`函数的值，`polygamma_backward`函数则用于计算梯度。实际的计算是在`polygamma_stub`函数中进行的，该函数的实现采用了CPU和CUDA两个后端，并使用了不同的算法。完整的`polygamma`函数实现涉及到多个文件和函数，包括`polygamma_stub`、`polygamma_cpu_kernel`、`polygamma_cuda_kernel`等。



更进一步来说，在CPU上，polygamma在`torch/csrc/autograd/Functions.cpp`文件中实现[(参考链接)](https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/function.cpp)
具体实现代码如下：
```
Tensor polygamma(const Tensor& input, int64_t n) {
  auto iter = TensorIterator::unary_op(input);
  polygamma_stub(iter.device_type(), iter, n);
  return iter.output();
}

DEFINE_DISPATCH(polygamma_stub);
```
其中，`polygamma_stub`是一个分派函数，它根据输入张量的设备类型和数据类型来调用不同的计算polygamma的函数。

而在CUDA上，polygamma在`torch/csrc/cuda/ElementWise.cu`文件中实现[(参考链接)](https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/function.cpp)
其实现逻辑是通过递归方式实现polygamma的计算
具体代码如下：
```
template <typename scalar_t>
struct PolyGammaOp {
  __device__ __forceinline__ scalar_t operator()(scalar_t x, int64_t n) const {
    scalar_t res = 0;
    scalar_t d = pow(x, n + 1);
    for (int64_t i = 0; i < n; ++i) {
      res += 1 / (x + i);
    }
    res = (-1) * pow(-1, n) * factorial(n - 1) * res / d;
    return res;
  }
};

void polygamma_kernel_cuda(TensorIteratorBase& iter, int64_t n) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "polygamma_cuda", [&] {
    gpu_kernel(iter, PolyGammaOp<scalar_t>{}, n);
  });
}

Tensor polygamma_cuda(const Tensor& self, int64_t n) {
  auto iter = TensorIterator::unary_op(self);
  polygamma_kernel_cuda(iter, n);
  return iter.output();
}
```
其中，`calc_polygamma`是一个计算polygamma的辅助函数，`factorial`是计算阶乘的辅助函数。在CUDA中，`PolyGammaOp`是一个functor，用于计算polygamma。`gpu_kernel`是一个PyTorch封装的CUDA kernel，用于在GPU上并行地计算polygamma。

## 2. MatLab

在 MatLab 中使用的 API 格式如下：

`Y = psi(k,x)`[(参考API文档)](https://ww2.mathworks.cn/help/matlab/ref/psi.html)

`Y = psi(X)` 为数组 X 的每个元素计算 digamma 函数。而polygamma函数则代表的是在 X 处的 digamma 函数的 k 阶导数。

## 3. Tensorflow

在 Tensorflow 中使用的 API 格式如下：

`tf.math.polygamma(a, x, name=None)`[(参考API文档)](https://www.tensorflow.org/api_docs/python/tf/math/polygamma)

上述函数参数中，a是一个非负值的张量；x是一个与a的d类型相同的张量；name定义了该操作的名称。

实现的代码如下：

```cpp
template <typename T>
class PolygammaOp : public UnaryElementWiseOp<T, PolygammaOp<T>> {
 public:
  using UnaryElementWiseOp<T, PolygammaOp<T>>::UnaryElementWiseOp;

  void Operate(OpKernelContext* context, const Tensor& input_tensor,
               Tensor* output_tensor) const override {
    const auto input = input_tensor.flat<T>();
    auto output = output_tensor->flat<T>();

    const int32 n = static_cast<int32>(this->param_);
    const int64 size = input.size();

    for (int64 i = 0; i < size; ++i) {
      output(i) = math::polygamma(n, input(i));
    }
  }
};

```

在该代码中，`PolygammaOp` 继承了 `UnaryElementWiseOp` 类，并实现了 `Operate` 方法，该方法实现了对 `tf.math.polygamma` 函数的操作。

## 4. Scipy

在 Scipy 中使用的 API 格式如下：

`scipy.special.polygamma(n, x)`[(参考API文档)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.polygamma.html)

其中，`n`表示要计算的 polygamma 函数的阶数；`x`表示 polygamma 函数的自变量。

实现的代码如下：

```cpp
static PyObject *polygamma_wrap(PyObject *self, PyObject *args)
{
    double x;
    int n;
    double res;

    if (!PyArg_ParseTuple(args, "di:polygamma", &n, &x))
        return NULL;

    if (n == 0) {
        res = digamma(x);
    } else if (n == 1) {
        res = trigamma(x);
    } else {
        res = psi(n, x);
    }

    return PyFloat_FromDouble(res);
}
```

在该代码中，`polygamma_wrap` 函数实现了 `scipy.special.polygamma` 函数的底层计算逻辑。当 `n` 等于 0 或 1 时，分别调用 `digamma` 和 `trigamma` 函数计算对应的 polygamma 值。当 `n` 大于 1 时，调用 `psi` 函数计算 n 阶 polygamma 值。

# 四、对比分析

## 1. 不同框架API使用方式

### 1. Scipy

```Python
import scipy.special

# 计算 digamma 函数在 x=5 处的值
x = 5
n = 0
result = scipy.special.polygamma(n, x)
print(f"digamma({x}) = {result}")

# 计算 trigamma 函数在 x=2.5 处的值
x = 2.5
n = 1
result = scipy.special.polygamma(n, x)
print(f"trigamma({x}) = {result}")

# 计算 pentagamma 函数在 x=3.3+1.2j 处的值
x = 3.3 + 1.2j
n = 4
result = scipy.special.polygamma(n, x)
print(f"pentagamma({x}) = {result}")
```

### 2. MatLab

```matlab
n = 2;
x = 3;

result = psi(n, x);
fprintf("Polygamma value for n = %d and x = %d: %f\n", n, x, result);
```

### 3. PyTorch

```Python
import torch

n = 2
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

result = torch.polygamma(n, x)
print("Polygamma values for n = {} and x = {}: {}".format(n, x.numpy(), result.numpy()))
```

### 4. Tensorflow

```Python
import tensorflow as tf

n = 2
x = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)

result = tf.math.polygamma(n, x)
print("Polygamma values for n = {} and x = {}: {}".format(n, x.numpy(), result.numpy()))
```

上述框架从使用体验来说，**Scipy**调用API方便，易于使用，但不支持输入为张量。**MatLab**中的polygamma值计算与digamma复用同一函数，实现逻辑相似但不支持输入张量。**PyTorch**和**Tensorflow** 支持输入张量，输出结果为一个 TensorFlow 的`tf.Tensor` 对象，如有需要可以借助其他方法转换为python标准类型。

# 五、设计思路与实现方案

## 命名与参数设计

<!-- 参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html) -->

API设计为 `paddle.polygamma(x, n, name)`。
其中 `x` 为非负值张量，允许的数据类型是float32和float64。`n` 表示多项式 gamma 函数的导数阶数，其为一个与 `x` 数据类型相同的张量。当`n=0`时，polygamma退化为 digamma。`name` 作为可选参数，定义了该操作的名称。

## 底层OP设计

polygamma可基于现有API即digamma进行实现，不再单独设计OP。

## API实现方案

该 API 实现于 `python/paddle/tensor/math.py`，通过调研发现，Paddle 本身已实现 paddle.digamma，可以计算gamma 函数的对数的一阶导数，可利用paddle.digamma API 做n阶导实现 paddle.polygamma。
具体来说，polygamma函数的k阶定义为：
```
polygamma(k, x) = d^k / dx^k [ln(gamma(x))]
```
根据导数的链式法则，可以将上述公式递归表示为：
```
polygamma(k, x) = polygamma(1, polygamma(k-1, x))
```
因此，初步设计的polygamma实现伪代码如下：
```
def digamma(x):
    # Digamma API implementation
    ...

def polygamma(x, k):
    if k == 1:
        return digamma(x)
    else:
        return (-1) ** (k - 1) * factorial(k - 1) * polygamma(x, k - 1) + digamma(x)
```

另外，由于该API需要考虑动静统一问题，故需要验证其在静态图中能否正常工作。

# 六、测试和验收的考量

<!-- 参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) -->

可考虑以下场景：

### 1. 一般场景
* 结果一致性测试。测试对于同一输入和 Tensorflow 以及 PyTorch 中polygamma API计算结果的数值的一致性。
* 数据类型测试。选取不同数据类型的输入，测试计算结果的准确性。
* 参数取值测试。选取不同取值的参数 `k` （表示求导的阶数），测试计算结果的准确性。
### 2. 边界条件
* 当 `x` 为空张量，测试其输出是否空张量且输出张量形状是否正确。
* 当 `n=0` ,测试其输出是否与digamma API得到的计算结果相同。
### 3. 异常测试
* 对于参数异常值输入，例如x的不合法值等，应该有友好的报错信息及异常反馈，需要有相关测试Case验证。

# 七、可行性分析和排期规划

本 API 主要参考已有API实现，难度适中，工期上能满足要求。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释

# 附件及参考资料
