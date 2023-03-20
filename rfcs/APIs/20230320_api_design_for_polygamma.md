# paddle.polygamma 设计文档

|API名称 | paddle.polygamma | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 吃点儿好的(https://github.com/KateJing1212/Paddle) | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-03-20 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20200320_api_design_for_polygamma.md | 


# 一、概述
## 1、相关背景

对于输入张量，其 digamma 函数的 n 阶导，称为多伽马函数`polygamma`。

## 2、功能目标

为 Paddle 新增 polygamma API，用于对输入张量的digamma函数进行n阶导操作。

## 3、意义

为 Paddle 新增 polygamma API，用于对输入张量的digamma函数进行n阶导操作。

# 二、飞桨现状

对飞桨框架目前不支持此功能，但支持 digamma 函数[(参考API文档)](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/digamma_cn.html)，且支持输入为张量。digamma 函数是 gamma 函数的对数的一阶导数，k 阶的 polygamma 函数是 gamma 函数的对数的 (k + 1) 阶导数。

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

## 2. MatLab

在 MatLab 中使用的 API 格式如下：

`Y = psi(k,x)`[(参考API文档)](https://ww2.mathworks.cn/help/matlab/ref/psi.html)

`Y = psi(X)` 为数组 X 的每个元素计算 digamma 函数。而polygamma函数则代表的是在 X 处的 digamma 函数的 k 阶导数。

## 3. Tensorflow

在 Tensorflow 中使用的 API 格式如下：

`tf.math.polygamma(a, x, name=None)`[(参考API文档)](https://www.tensorflow.org/api_docs/python/tf/math/polygamma)

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

API设计为 `paddle.Tensor.polygamma(n, x)`。其中 `x` 为 `B1 X ... X Bn X P X M` 张量，`n` 表示多项式 gamma 函数的导数阶数。当`n=0`时，polygamma退化为 digamma。

## 底层OP设计

polygamma可基于现有API即digamma进行实现，不再单独设计OP。

## API实现方案

该 API 实现于 python/paddle/tensor/math.py，通过调研发现，Paddle 本身已实现 paddle.digamma，可以计算gamma 函数的对数的一阶导数，可利用paddle.digamma API 做n阶导实现 paddle.polygamma。 而 Paddle 中已有 paddle.digamma API 的具体实现逻辑，位于 python/paddle/tensor/math.py 下的 digamma 函数中。

# 六、测试和验收的考量

<!-- 参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html) -->

可考虑一下场景：

1. 当`x` 为空张量，输出为空张量，且输出张量形状正确；
2. 结果一致性，和 Tensorflow 以及 PyTorch 结果的数值的一致性, `paddle.polygamma(n=5, x)` , `torch.polygamma(5, x))` 和 `tf.math.polygamma(5, x)` 结果是否一致；
3. 异常测试，对于参数异常值输入，应该有友好的报错信息及异常反馈，需要有相关测试Case验证。

# 七、可行性分析和排期规划

本 API 主要参考已有API实现，难度适中，工期上能满足要求。

# 八、影响面

为独立新增API，对其他模块没有影响。

# 名词解释

# 附件及参考资料
