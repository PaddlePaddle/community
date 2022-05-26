# paddle.logspace设计文档


|API名称 | paddle.logspace |
|---|---|
|提交作者 | 为往圣继绝学 |
|提交时间 | 2022-03-21 |
|版本号 | V1.0 |
|依赖飞桨版本 | develop |
|文件名 | 20220321_api_design_for_logspace.md |


# 一、概述
## 1、相关背景
`logspace`与`linspace`类似，`linspace(a, b, n)`产生了一个以`a`为首项、以`b`为末项、共有`n`项的等差数列，现在我们希望由`logspace(a, b, n, m)`产生一个以`m**a`为首项、以`m**b`为末项、共有`n`项的等比数列。

## 2、功能目标

在飞桨中增加`paddle.logspace`这个API。

## 3、意义

飞桨将直接提供`logspace`这个API。

# 二、飞桨现状
飞桨目前没有直接提供此API，但是可以通过组合API的方式实现：

```python
def logspace(start, end, num, base=10.0):
    y = paddle.linspace(start, end, num)
    x = paddle.full_like(y, base)
    return paddle.pow(x, y)
```


# 三、业内方案调研
在百度中搜索“pytorch logsapce”、“numpy logspace”和“tensorflow logspace”，发现PyTorch和Numpy中都有logspace这个API，而TensorFlow中还没有。

## PyTorch

### 实现解读

在PyTorch中，logspace是由C++实现的，核心代码为：

```c++
Tensor& logspace_out(const Scalar& start, const Scalar& end, int64_t steps, double base, Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }

  if (result.device() == kMeta) {
    return result;
  }

  Tensor r = result.is_contiguous() ? result : result.contiguous();

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    if (isComplexType(r.scalar_type())){
      r.fill_(std::pow(base, start.to<c10::complex<double>>()));
    } else {
      r.fill_(std::pow(base, start.to<double>()));
    }
  } else if (isComplexType(r.scalar_type())) {
    AT_DISPATCH_COMPLEX_TYPES(r.scalar_type(), "logspace_cpu", [&]() {
      scalar_t scalar_base = static_cast<scalar_t>(base);
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t *data_ptr = r.data_ptr<scalar_t>();
      scalar_t step = (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
      const int64_t halfway = steps / 2;
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        scalar_t is = static_cast<scalar_t>(p_begin);
        for (int64_t i = p_begin; i < p_end; ++i, is+=1) { //std::complex does not support ++operator
          if (i < halfway) {
            data_ptr[i] = std::pow(scalar_base, scalar_start + step*is);
          } else {
            data_ptr[i] = std::pow(scalar_base, scalar_end - (step * static_cast<scalar_t>(steps - i - 1)));
          }
        }
      });
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND(kBFloat16, r.scalar_type(), "logspace_cpu", [&]() {
      double scalar_base = static_cast<double>(base); // will be autopromoted anyway
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      scalar_t *data_ptr = r.data_ptr<scalar_t>();
      double step = static_cast<double>(scalar_end - scalar_start) / (steps - 1);
      const int64_t halfway = steps / 2;
      at::parallel_for(0, steps, internal::GRAIN_SIZE, [&](int64_t p_begin, int64_t p_end) {
        for (const auto i : c10::irange(p_begin, p_end)) {
          if (i < halfway) {
            data_ptr[i] = std::pow(scalar_base, scalar_start + step*i);
          } else {
            data_ptr[i] = std::pow(scalar_base, scalar_end - step * (steps - i - 1));
          }
        }
      });
    });
  }

  if (!result.is_contiguous()) {
    result.copy_(r);
  }
  return result;
}
```
上面代码的行为是：

1. 检查输入（steps非负）；
2. 分情况处理：
    - 如果steps=0，直接返回空张量；
    - 如果steps=1，直接返回形为`[1]`的张量；
    - 如果steps>1，再往下执行：
3. 根据结果类型是否为复数，分别计算：指数的步长`step=(end - start)/(steps - 1)`，然后
    - 当指标i<steps/2时计算`base**(start+i*step)`，
    - 当指标i>=steps/2时计算`base**(end-(steps-i-1)*step)`。

### 使用示例

```python
>>> import torch
>>> torch.logspace(1, 10, 4, 2)
tensor([   2.,   16.,  128., 1024.])
>>> torch.logspace(1, 10j, 4, 2)
tensor([ 2.0000+0.0000j, -1.0700+1.1726j, -0.1150-1.2547j,  0.7971+0.6038j])
```

## NumPy

### 实现解读

在NumPy中，logspace是在Python层面借助linspace实现的，代码为：

```python
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    if dtype is None:
        return _nx.power(base, y)
    return _nx.power(base, y).astype(dtype, copy=False)
```

其行为就是将linspace的结果逐元素取以`base`为底的指数。

此外，比PyTorch更丰富的是，`start`和`end`可以是数组，此时还能用`axis`决定等比数量所沿着的轴。最后，还能用`endpoint`指定`stop`这个区间端点是开的还是闭的。

### 使用示例

```python
>>> import numpy
>>> numpy.logspace(1, 10, 4, 2)
array([1.e+01, 1.e+04, 1.e+07, 1.e+10])
>>> numpy.logspace([1, 2], [10, 20], 4, 2)
array([[1.e+01, 1.e+02],
       [1.e+04, 1.e+08],
       [1.e+07, 1.e+14],
       [1.e+10, 1.e+20]])
>>> numpy.logspace([1, 2], [10, 20], 4, 2, axis=1)
array([[1.e+01, 1.e+04, 1.e+07, 1.e+10],
       [1.e+02, 1.e+08, 1.e+14, 1.e+20]])
```

# 四、对比分析
- `numpy.logspace`比`torch.logspace`功能更加丰富，支持高维的`start`和`end`，而`torch.logspace`只支持标量输入。
- 在细节上，`numpy.logspace`可以决定生成的数列是否包含`end`，而`torch.logspace`不能。
- `numpy.logspace`和`torch.logspace`都支持复数。

对于`paddle.logspace`，我们参考`paddle.linspace`将其设计为：

- 只支持标量的`start`和`end`；
- 不能决定结果是否包含`end`；
- 暂时不支持复数。

# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.logspace(start, stop, num, base=10.0, dtype=None, name=None)`，它能产生一个以`base**start`为首项、以`base**end`为末项、共有`num`项的等比数列。

参数类型要求：

- `start`、`end`、`base`可以是`int`、`float`或者形状为`[1]`的`Tensor`，该`Tensor`的数据类型可以是`float32`、`float64`、`int32` 或`int64`；
- `num`是一个整数或者类型为`int32`、形状为`[1]`的`Tensor`；
- `dtype`是输出`Tensor`的数据类型，可以是`float32`、`float64`、 `int32`或`int64`。如果`dtype`是`None`，输出`Tensor`数据类型为`float32`。

## 底层OP设计

logspace算子的描述添加在`paddle/fluid/operators/logspace_op.cc`。

在`paddle/phi/infermeta/multiary.h`中声明形状推断的函数原型：

```c++
void LogspaceInferMeta(const MetaTensor& start,
                       const MetaTensor& stop,
                       const MetaTensor& number,
                       const MetaTensor& base,
                       MetaTensor* out);
```

其实现位于`paddle/phi/infermeta/multiary.cc`。

核函数的原型（位于`paddle/phi/kernels/logspace_kernel.h`）设计为

```c++
template <typename T, typename Context>
void LogspaceKernel(const Context& ctx,
                    const DenseTensor& start,
                    const DenseTensor& stop,
                    const DenseTensor& number,
                    const DenseTensor& base,
                    DataType dtype,
                    DenseTensor* out);
```

核函数在CPU和GPU的实现与注册分别位于：

- `paddle/phi/kernels/cpu/logspace_kernel.cc`
- `paddle/phi/kernels/gpu/logspace_kernel.cu`

## API实现方案

在`python/paddle/fluid/layers/tensor.py`中增加`logspace`函数：

```python
def logspace(start, stop, num, base=10.0, dtype=None, name=None):
    # ...
    # 参数检查
    # ...
    if in_dygraph_mode():
        return _C_ops.logspace(tensor_start, tensor_stop, tensor_num, tensor_base, 'dtype', dtype)
    # ...
    # 静态图代码
    # ...
```

# 六、测试和验收的考量

- 输入合法性检验；

- 对比与Numpy的结果的一致性：

  - 当start<end时，

  - 当end<start时，

  - 当num=1时，

  - 当num不是整数、正数时，

  - 当base>0时，

  - 当base=0时，

  - 当base<0时；

- 对各种`dtype`的测试；
- 动态图、静态图测试；
- CPU、GPU测试。

# 七、可行性分析和排期规划
已经基本完成，在该文档通过验收后可快速提交。

# 八、影响面
logspace是独立API，不会对其他API产生影响。

# 名词解释
无

# 附件及参考资料
无
