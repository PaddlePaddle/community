# paddle.heaviside设计文档

|API名称 | paddle.heaviside |
|---|---|
|提交作者 | 为往圣继绝学 |
|提交时间| 2022-03-22 |
|版本号 | V1.0 |
|依赖飞桨版本| develop |
|文件名 | 20220322_api_design_for_heaviside.md |


# 一、概述
## 1、相关背景
heaviside指赫维赛德阶跃函数，它的表达式为
```
           /
          | 0, x < 0
h(x, y) = < y, x = 0
          | 1, x > 0
           \
```

它的两个偏导数为

```
             /
            | 0, x != 0
h_x(x, y) = <
            | ∞, x = 0
             \
```
和
```
             /
            | 0, x != 0
h_y(x, y) = <
            | 1, x = 0
             \
```

## 2、功能目标

在飞桨中增加赫维赛德阶跃函数。

## 3、意义

飞桨将直接提供赫维赛德阶跃函数。

# 二、飞桨现状
飞桨目前没有提供heaviside这个API，但是可以通过组合API的方式实现：

```python
def heaviside(x, y):
    out = 0 * (x < 0).cast(x.dtype) + \
          y * (x == 0).cast(x.dtype) + \
          1 * (x > 0).cast(x.dtype)
    return out
```

另外，在`paddle/fluid/operators/margin_rank_loss_op.h`中一段代码：

```c++
template <typename T>
struct Heaviside {
  HOSTDEVICE T operator()(const T& val) const {
    return static_cast<T>(val > 0 ? 1 : 0);
  }
};
```

它是只能取0或1的赫维赛德阶跃函数。

# 三、业内方案调研

## PyTorch

### 实现解读

PyTorch中的heaviside是用c++实现的，核心代码为：
```c++
void heaviside_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kHalf, kBool, kBFloat16, iter.dtype(), "heaviside_cpu", [&]() {
    cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
        return a == 0 ? b : static_cast<scalar_t>(a > 0);
    });
  });
}
```

它的逻辑是先判断a是否为零，如果是零，就直接回b，如果不是零，就把(a>0)这个真假值类型转换成标量，刚好是0或1。

### 使用示例

```python
>>> import torch
>>> x = torch.linspace(-1, 1, 5)
>>> x
tensor([-1.0000, -0.5000,  0.0000,  0.5000,  1.0000])
>>> torch.heaviside(x, torch.Tensor([2]))
tensor([0., 0., 2., 1., 1.])
>>> y = torch.stack([torch.linspace(-1, 1, 5), torch.linspace(-2, 2, 5)], 1)
>>> y
tensor([[-1.0000, -2.0000],
        [-0.5000, -1.0000],
        [ 0.0000,  0.0000],
        [ 0.5000,  1.0000],
        [ 1.0000,  2.0000]])
>>> torch.heaviside(y, torch.Tensor([5, 6]))
tensor([[0., 0.],
        [0., 0.],
        [5., 6.],
        [1., 1.],
        [1., 1.]])
>>> y.requires_grad = True
>>> z = torch.heaviside(y, torch.Tensor([5, 6])).sum()
>>> z.backward()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "path/to/lib/python3.8/site-packages/torch/_tensor.py", line 363, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
  File "path/to/lib/python3.8/site-packages/torch/autograd/__init__.py", line 173, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: derivative for aten::heaviside is not implemented
```

通过实践验证，我们发现PyTorch中的heaviside是支持广播的，但是不能反向传播梯度。

## NumPy

### 实现解读

我没有找到NumPy中heaviside的完整源代码，只找到了下面这样一段疑似源代码的东西：

```
@type@ npy_heaviside@c@(@type@ x, @type@ h0)
{
    if (npy_isnan(x)) {
        return (@type@) NPY_NAN;
    }
    else if (x == 0) {
        return h0;
    }
    else if (x < 0) {
        return (@type@) 0.0;
    }
    else {
        return (@type@) 1.0;
    }
}
```

它使用了if-else语句，根据三种情况分别返回对应的值。

### 使用示例

```python
>>> import numpy
>>> x = numpy.linspace(-1, 1, 5)
>>> x
array([-1. , -0.5,  0. ,  0.5,  1. ])
>>> numpy.heaviside(x, 2)
array([0., 0., 2., 1., 1.])
>>> y = numpy.linspace([-1, -2], [1, 2], 5, axis=0)
>>> y
array([[-1. , -2. ],
       [-0.5, -1. ],
       [ 0. ,  0. ],
       [ 0.5,  1. ],
       [ 1. ,  2. ]])
>>> numpy.heaviside(y, [5, 6])
array([[0., 0.],
       [0., 0.],
       [5., 6.],
       [1., 1.],
       [1., 1.]])
```

通过实践验证，我们发现NumPy中的heaviside也是支持广播的。

## TensorFlow

### 实现解读

在TensorFlow中，heaviside的调用路径为`tensorflow.experimental.numpy.heaviside`，其实现代码为

```python
@np_utils.np_doc('heaviside')
def heaviside(x1, x2):  # pylint: disable=missing-function-docstring

  def f(x1, x2):
    return array_ops.where_v2(
        x1 < 0, constant_op.constant(0, dtype=x2.dtype),
        array_ops.where_v2(x1 > 0, constant_op.constant(1, dtype=x2.dtype), x2))

  y = _bin_op(f, x1, x2)
  if not np.issubdtype(y.dtype.as_numpy_dtype, np.inexact):
    y = y.astype(np_dtypes.default_float_type())
  return y
```
从代码中可以看到，TensorFlow中的heaviside是分别找到数组中小于0、大于0和等于0的元素的位置，然后分别填充为0、1和给定的数。

### 使用示例

```python
>>> import tensorflow
>>> tensorflow.linspace(-1, 1, 5)
<tf.Tensor: shape=(5,), dtype=float64, numpy=array([-1. , -0.5,  0. ,  0.5,  1. ])>
>>> x = tensorflow.linspace(-1, 1, 5)
>>> x
<tf.Tensor: shape=(5,), dtype=float64, numpy=array([-1. , -0.5,  0. ,  0.5,  1. ])>
>>> tensorflow.experimental.numpy.heaviside(x, 2)
<tf.Tensor: shape=(5,), dtype=float64, numpy=array([0., 0., 2., 1., 1.])>
```

由于我本身并不熟悉TensorFlow，就先不对其广播机制和梯度传播进行验证了（从文档上看，它应该支持广播）。

# 四、对比分析

在功能上，PyTorch、NumPy以及TensorFlow中的heaviside是一致的，且都支持广播。遗憾的是PyTorch没有为heaviside设计梯度。

在底层设计上，PyTorch使用了三目运算符，代码简洁；NumPy使用if-else语言，虽然代码冗长但是逻辑清晰；TensorFlow采用了查找-填充的方式，逻辑合理，但效率偏低。

因此，我们计划在飞桨中设计一个使用三目运算符来计算的、支持广播的、具有梯度的heaviside算子。

# 五、设计思路与实现方案

- 前向计算设计为`x == 0 ? y : static_cast<T>(x > 0) `；
- 关于`x`的偏导数设计为恒等于0（与事实上的偏导数仅在直线x=0上取值不同，事实上此时偏导数是∞）；
- 关于`y`的偏导数设计为`static_cast<T>(x == 0) `。

## 命名与参数设计

API设计为`paddle.heaviside(x, y, name=None)`，它逐元素计算输入的两个Tensor的heaviside函数，支持广播，参数为

- x （Tensor）- 输入的Tensor。数据类型为 float32、 float64、int32或 int64；
- y （Tensor）- 输入的Tensor。数据类型为 float32、 float64 、int32或int64；
- name （str, 可选）- 操作的名称(可选，默认值为None）。

也可以通过`paddle.Tensor.heaviside(y)`来调用。

## 底层OP设计

在`paddle/fluid/operators/elementwise/elementwise_heaviside_op.cc`中增加heaviside算子的描述。

### 正向算子

在`paddle/phi/kernels/funcs/elementwise_functor.h`增加heaviside的函子：

```c++
template <typename T>
struct HeavisideFunctor {
  inline HOSTDEVICE T operator()(const T a, const T b) const {
    return a == static_cast<T>(0) ? b : static_cast<T>(a > static_cast<T>(0));
  }
};
```

在`paddle/phi/kernels/impl/elementwise_kernel_impl.h`利用heaviside的函子完成heaviside算子的核函数的实现：

```c++
template <typename T, typename Context>
void ElementwiseHeavisideKernel(const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& y,
                                int axis,
                                DenseTensor* out) {
    // ...
}
```

### 反向算子

在`paddle/phi/kernels/funcs/elementwise_functor.h`增加heaviside偏导数的函子：

```c++
template <typename T>
struct HeavisideGradDx {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(0);
  }
};

template <typename T>
struct HeavisideGradDy {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return dout * static_cast<T>(x == static_cast<T>(0));
  }
};
```

在`paddle/phi/kernels/impl/elementwise_kernel_impl.h`利用heaviside导数的函子完成heaviside算子的反向核函数的实现：

```c++
template <typename T, typename Context>
void ElementwiseHeavisideGradKernel(const Context& dev_ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& y,
                                    const DenseTensor& out_grad,
                                    int axis,
                                    DenseTensor* x_grad,
                                    DenseTensor* y_grad) {
    // ...
}
```

## API实现方案

在`python/paddle/tensor/math.py`中增加：
```python
def heaviside(x, y, name=None):
    op_type = 'elementwise_heaviside'
    axis = -1
    act = None
    if paddle.in_dynamic_mode():
        return _elementwise_op_in_dygraph(
            x, y, axis=axis, act=act, op_name=op_type)
    return _elementwise_op(LayerHelper(op_type, **locals()))
```

# 六、测试和验收的考量

- 输入合法性检验；
    - 输入不是张量，
    - 输入的dtype不一致；
- 与Numpy对比计算结果的一致性：
    - x和y是形状都是[13, 17]，
    - x的形状是[2, 3, 20]，y的形状是[1]，
    - x的形状是[100, 5, 2]，y的形状是[100, 1, 1]，
    - x的形状是[2, 100, 3]，y的形状是[100, 1]，
    - x的形状是[1, 3, 100]，y的形状是[100]，
    - x的形状是[2, 50, 2, 1]，y的形状是[50, 2, 1]，
    - x的形状是[2, 3, 4, 5]，y的形状是[2, 3, 1, 5]；
- 梯度测试；
- 对各种`dtype`的测试；
- 动态图、静态图测试；
- CPU、GPU测试。

# 七、可行性分析和排期规划
已经基本完成，待该设计文档通过验收后可快速提交。

# 八、影响面
可能的争议：把赫维赛德阶跃函数h(x, y)的偏导数h_x(x,y)设计为恒等于0与数学事实不符，但这在实际计算中会带来便利。

对其他模块的影响：无。

# 名词解释

奥利弗·赫维赛德（Oliver Heaviside，1850年5月18日－1925年2月3日），英国自学成才的物理学家，出生于伦敦卡姆登镇。——《百度百科》

# 附件及参考资料

无
