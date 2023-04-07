# paddle.polar 设计文档

| API 名称     |                paddle.polar           |
| ------------ | ---------------------------------------- |
| 提交作者     | PommesPeter                               |
| 提交时间     | 2023-02-21                                |
| 版本号       | V1.0                                      |
| 依赖飞桨版本  | develop                                   |
| 文件名       | 20230221_api_design_for_polar.md      |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持科学计算相关 API，Paddle 需要扩充 API `paddle.polar`。

## 2、功能目标

增加 API `paddle.polar`，通过输入模和相位角，`elementwise` 构造复数 tensor。方便计算极坐标系下的运算。

## 3、意义

为 Paddle 增加极坐标和复数的计算函数，丰富 `paddle` 中科学计算相关的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 `polar` API，但是存在 `paddle.complex`，参考其他框架可以发现，Paddle 没有专门针对极坐标系下进行计算的 api，无法构建极坐标，直接使用 `paddle.complex` 代码不够清晰易读。
- 该 API 的实现及测试主要参考目前 Paddle 中含有的 `paddle.complex`。

# 三、业内方案调研

## PyTorch

PyTorch 中有 `torch.polar` 的API，详细参数为 `torch.polar(abs, angle, *, out=None) → Tensor`。

在 PyTorch 中的介绍为：

> Constructs a complex tensor whose elements are Cartesian coordinates corresponding to the polar coordinates with absolute value `abs` and angle `angle`.

在实现方法上，PyTorch 是通过 C++ API 组合实现的，[代码位置](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/TensorFactories.cpp#L190-L251)

实现代码：

```cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complex / polar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void complex_check_floating(const Tensor& a, const Tensor& b) {
  TORCH_CHECK((a.scalar_type() == kFloat || a.scalar_type() == kDouble || a.scalar_type() == kHalf) &&
              (b.scalar_type() == kFloat || b.scalar_type() == kDouble || b.scalar_type() == kHalf),
              "Expected both inputs to be Half, Float or Double tensors but got ",
              a.scalar_type(), " and ", b.scalar_type());
}

void complex_check_dtype(
    const Tensor& result,
    const Tensor& a,
    const Tensor& b) {
  complex_check_floating(a, b);
  TORCH_CHECK(a.scalar_type() == b.scalar_type(),
              "Expected object of scalar type ", a.scalar_type(),
              " but got scalar type ", b.scalar_type(), " for second argument");
  TORCH_CHECK(result.scalar_type() == toComplexType(a.scalar_type()),
              "Expected object of scalar type ", toComplexType(a.scalar_type()),
              " but got scalar type ", result.scalar_type(),
              " for argument 'out'");
}

Tensor& complex_out(const Tensor& real, const Tensor& imag, Tensor& result) {
  complex_check_dtype(result, real, imag);
  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_input(real)
      .add_input(imag)
      .check_all_same_dtype(false)
      .build();
  complex_stub(iter.device_type(), iter);
  return result;
}

Tensor complex(const Tensor& real, const Tensor& imag) {
  complex_check_floating(real, imag);
  c10::TensorOptions options = real.options();
  options = options.dtype(toComplexType(real.scalar_type()));
  Tensor result = at::empty(0, options);
  return at::complex_out(result, real, imag);
}

Tensor& polar_out(const Tensor& abs, const Tensor& angle, Tensor& result) {
  complex_check_dtype(result, abs, angle);
  auto iter = TensorIteratorConfig()
      .add_output(result)
      .add_input(abs)
      .add_input(angle)
      .check_all_same_dtype(false)
      .build();
  polar_stub(iter.device_type(), iter);
  return result;
}

Tensor polar(const Tensor& abs, const Tensor& angle) {
  complex_check_floating(abs, angle);
  c10::TensorOptions options = abs.options();
  options = options.dtype(toComplexType(abs.scalar_type()));
  Tensor result = at::empty(0, options);
  return at::polar_out(result, abs, angle);
}
}
```

参数表：

- abs：复数张量的绝对值。必须为 float 或 double。
- angle：复数张量的角度。数据类型必须与abs相同。
- out：如果输入为 torch.float32，则必须为 torch.complex64。如果输入为 torch.float64，则必须为 torch.complex128。

# 四、对比分析

## 共同点

- 都能通过输入模和相位角，`elementwise` 构造复数 tensor。方便计算极坐标系下的运算。

## 不同点

- PyTorch 是在 C++ API 基础上实现，使用 Python 调用 C++ 对应的接口。

# 五、设计思路与实现方案

## 命名与参数设计

添加 API

```python
paddle.polar(
    abs: Tensor,
    angle: Tensor,
    name: str=None
)
```

## 底层OP设计

使用已有的 API 组合实现，不再单独设计 OP。

需要注意：如果输入是 torch.float32，则必须是 torch.complex64。如果输入是 torch.float64，则必须是 torch.complex128。

## API实现方案

该 API 实现于 `python/paddle/tensor/creation.py`

通过调研发现，计算该极坐标可以使用复数计算，Paddle 本身已实现 `paddle.complex`，可利用已有 API 实现。代入公式：

$$
\text{out} = \text{abs}\cdot\cos(\text{angle}) + \text{abs}\cdot\sin(\text{angle})\cdot j
$$

即可得到对应模和相位角的极坐标以及所对应的笛卡尔坐标。

随后，Paddle 中已有 `complex` API 的具体实现逻辑，位于 `python/paddle/tensor/creation.py` 下的 `complex` 函数中，因此只需要调用其函数构造复数即可。

# 六、测试和验收的考量

测试需要考虑的 case 如下：

- 输出数值结果的一致性和数据类型是否正确，使用 pytorch 作为参考标准
- 参数 `abs` 的数据类型准确性判断
- 参数 `angle` 的数据类型准确性判断、

# 七、可行性分析和排期规划

方案主要依赖现有 Paddle API 组合而成，且依赖的 `paddle.complex` 已经在 Paddle repo 的 [python/paddle/tensor/creation.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py#L2160-L2209)。工期上可以满足在当前版本周期内开发完成。

# 八、影响面

新增 API，对其他模块无影响

# 名词解释

无

# 附件及参考资料

[torch.polar](https://pytorch.org/docs/stable/generated/torch.polar.html)

[scipy.linalg.polar](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.polar.html)

[paddle.complex](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/tensor/creation.py#L2160-L2209)