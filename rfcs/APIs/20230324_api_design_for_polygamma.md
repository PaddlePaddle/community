# paddle.polygamma 设计文档

|API名称 | paddle.polygamma | 
|---|---|
|提交作者 | PommesPeter | 
|提交时间 | 2023-05-23 | 
|版本号 | V2.0 | 
|依赖飞桨版本 | develop | 
|文件名 | 20200324_api_design_for_polygamma.md | 


# 一、概述
## 1、相关背景

对于输入张量，其 digamma 函数的 n 阶导，称为多伽马函数`polygamma`。

## 2、功能目标

为 Paddle 新增 polygamma API，用于对输入张量的digamma函数进行n阶导操作。

## 3、意义

为 Paddle 新增 polygamma API，用于对输入张量的digamma函数进行n阶导操作。

# 二、飞桨现状

飞桨框架目前不支持此功能，但支持 digamma 函数[(参考API文档)](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/digamma_cn.html) ，且支持输入为张量。digamma 函数是 gamma 函数的对数的一阶导数，k 阶的 polygamma 函数是 gamma 函数的对数的 (k + 1) 阶导数。

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

上述函数参数中，`input` 是一个 Tensor，表示要计算polygamma函数的输入值；`n` 是一个整数，表示要计算的polygamma函数的阶数；`out` 是一个可选的输出Tensor，用于存储计算结果。实现的代码如下：

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
可以看到，`polygamma` 函数调用了 `polygamma_stub` 函数来计算 `polygamma` 函数的值，`polygamma_backward` 函数则用于计算梯度。实际的计算是在 `polygamma_stub` 函数中进行的。完整的`polygamma`函数实现涉及到多个文件和函数。

更进一步来说，在 CPU 上，polygamma在 `aten/src/ATen/native/cpu/UnaryOpsKernel.cpp` 文件中实现[(参考链接)](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp)
其计算逻辑是通过调用 digamma API 和 trigamma API 实现 polygamma 的计算，具体实现代码如下：

```cpp
static void polygamma_kernel(TensorIteratorBase& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel(iter);
  } else if (n == 1) {
    trigamma_kernel(iter);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.dtype(), "polygamma", [&]() {
      cpu_kernel(
          iter, [=](scalar_t a) -> scalar_t { return calc_polygamma(a, n); });
    });
  }
}
```

另外，引用到的 `calc_polygamma` 作为辅助函数，其实现代码如下：

```cpp
template <typename scalar_t, bool is_cuda=false>
static inline C10_HOST_DEVICE scalar_t calc_polygamma(scalar_t x, int n) {
  // already blocked if n <= 1
  const auto one = scalar_t{1};
  return ((n % 2) ? one : -one) *
      std::exp(std::lgamma(static_cast<scalar_t>(n) + one)) *
      zeta<scalar_t, is_cuda>(static_cast<scalar_t>(n + 1), x);
}
```

在 CUDA 上，polygamma 在`aten/src/ATen/native/cpu/UnaryOpsKernel.cpp` 文件中实现[(参考链接)](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/UnaryGammaKernels.cu)，具体实现代码如下：

```cpp
CONSTEXPR_EXCEPT_WIN_CUDA char polygamma_name[] = "polygamma";
void polygamma_kernel_cuda(TensorIteratorBase& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel_cuda(iter);
  } else if (n == 1) {
    trigamma_kernel_cuda(iter);
  } else {
#if AT_USE_JITERATOR()
    // TODO : `unary_jitted_gpu_kernel` for cleaner UX.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        iter.common_dtype(), "polygamma_cuda", [&]() {
          jitted_gpu_kernel<
              /*name=*/polygamma_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(
              iter,
              polygamma_string,
              /*scalar_pos=*/at::cuda::jit::BinaryFuncVariant::NoScalar,
              /*scalar_val=*/0,
              /*extra_args=*/std::make_tuple(n));
        });
#else
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        iter.common_dtype(), "polygamma_cuda", [&]() {
          gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return calc_polygamma<scalar_t, /*is_cuda=*/true>(a, static_cast<int>(n));
          });
        });
#endif // AT_USE_JITERATOR()
  }
}
```

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

上述框架从使用体验来说，**Scipy** 调用 API 方便，易于使用，但不支持输入为张量。**MatLab** 中的 polygamma 值计算与 digamma 复用同一函数，实现逻辑相似但不支持输入张量。**PyTorch** 和 **Tensorflow** 支持输入张量，输出结果为一个 `Tensor` 对象，如有需要可以借助其他方法转换为 python 标准类型。

# 五、设计思路与实现方案

## 命名与参数设计

API设计为 `paddle.polygamma(x, n, name=None)`。

- `x` 为张量，允许的数据类型是 float32 和 float64，其原因是 digamma 对数据类型进行了限制；`n` 表示多项式 gamma 函数的导数阶数，其可以为一个非负整数或一个非负值张量，允许的数据类型是 int32 和 int64；如果 `n` 是一个张量，需要保证 `n` 的维度 `ndim` 和 `n` 的参数量 `numel` 满足 `ndim <= 1` 且 `numel == 1`；

- 当 `n=0` 时，polygamma 退化为 digamma。

- `name`作为可选参数，定义了该操作的名称，其默认值为 `None`。
另外，该API还支持 `Tensor.polygamma(n)` 的调用形式。

## 底层OP设计

polygamma 中 `n=0` 的情况基于现有 API 即 digamma 进行实现，此外的情况将设计 `PolygammaKernel`。

## API实现方案

该 API 实现于 `python/paddle/tensor/math.py`，通过调研发现，Paddle 本身已实现 `pdadle.digamma`，可以计算 gamma 函数的对数的一阶导数，可利用 `paddle.digamma` API 做 n 阶导实现 `paddle.polygamma`。具体来说，polygamma 函数的 k 阶定义为：

$$\Phi^k(x) = \frac{d^k}{dx^k} [\ln(\Gamma(x))]$$

根据其[数学定义](https://zh.wikipedia.org/wiki/%E5%A4%9A%E4%BC%BD%E7%8E%9B%E5%87%BD%E6%95%B0)，polygamma 可以使用级数表示法表示为：

$$ \Phi^k(x) = (-1)^{k+1}k!\zeta(k + 1, x) $$

进一步可以化简为：

$$ \Phi^k(x) = (-1)^{k+1}k!\zeta(k + 1, x) = (-1)^{k+1}\Gamma(k + 1)\zeta(k + 1, x) $$

对于实现角度而言，可使用 C++ 标准库当中的 `std::lgamma(x)`，即表示为 $\ln(\Gamma(x))$，对其求自然指数即可得到 $e^{\ln\Gamma(x)}=\Gamma(x)$，所以实现角度而言可以转换为：

$$ \Phi^k(x) = (-1)^{k+1}\Gamma(k + 1)\zeta(k + 1, x) = (-1)^{k+1}e^{\text{lgamma}(x)}\zeta(k + 1, x)$$

$\zeta$ 函数的实现可参考 pytorch，方便用于计算。该实现不需要使用递归实现，计算性能较高。

因此，本设计中，默认 `k = 0` 表示 $\ln\Gamma(x)$ 的一阶导数，初步设计的 polygamma 实现伪代码如下：

```python
def polygamma(x, k):
    if k == 0:
        return digamma(x)
    else:
        # Running polygamma kernel code.
        ...
```

另外，由于该API需要考虑动静统一问题，故需要验证其在静态图中能否正常工作。

# 六、测试和验收的考量

可考虑以下场景：

1. 一般场景

- 结果一致性测试。测试对于同一输入和 Scipy 中 polygamma AP I计算结果的数值的一致性。
- 数据类型测试。选取不同数据类型的输入，测试计算结果的准确性。
- 参数取值测试。选取不同取值的参数 `k` （表示求导的阶数），测试计算结果的准确性。

1. 边界条件

- 当 `x` 为空张量，测试其输出是否空张量且输出张量形状是否正确。
- 当 `n=0` ,测试其输出是否与digamma API得到的计算结果相同。

3. 异常测试

- 对于参数异常值输入，例如x的不合法值等，应该有友好的报错信息及异常反馈，需要有相关测试 Case 验证。

# 七、可行性分析和排期规划

本 API 主要参考已有 API 实现，难度适中，工期上能满足要求。

# 八、影响面

为独立新增 API，对其他模块没有影响。

# 名词解释

# 附件及参考资料
