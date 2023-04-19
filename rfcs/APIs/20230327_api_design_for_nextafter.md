# paddle.nextafter 设计文档

| API 名称     |                paddle.nextafter           |
| ------------ | ---------------------------------------- |
| 提交作者     | enkilee                                   |
| 提交时间     | 2023-03-27                                |
| 版本号       | V1.0                                      |
| 依赖飞桨版本  | develop                                   |
| 文件名       | 20230327_api_design_for_nextafter.md      |

# 一、概述

## 1、相关背景

为了提升飞桨 API 丰富度，支持科学计算相关 API，Paddle 需要扩充 API `paddle.nextafter`。

## 2、功能目标

增加 API `paddle.polar`，将输入后的下一个浮点值返回给其他元素，输入和其他 shape 必须是可广播的。

## 3、意义

Paddle 将可以使用 `paddle.nextafter` 找到当前输入的下一个浮点值，丰富 `paddle` 中科学计算相关的 API。

# 二、飞桨现状

- 目前 Paddle 缺少 `nextafter` API，参考其他框架可以发现，Paddle 没有专门下一个浮点值进行计算的 api。

# 三、业内方案调研

## 3.1 PyTorch

PyTorch 中有 `torch.nextafter` 的API，详细参数为 `torch.nextafter(input, other, *, out=None) → Tensor`。

在 PyTorch 中的介绍为：

>Return the next floating-point value after input towards other, elementwise.The shapes of input and other must be broadcastable.

在实现方法上，PyTorch 是通过 std::nextafter 实现的，[代码位置](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cpu/vec/vec_base.h#L473-L479)
### 实现代码：

```cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ complex / polar ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Vectorized<T> nextafter(const Vectorized<T> &b) const {
    Vectorized<T> ret;
    for (const auto i : c10::irange(size())) {
      ret[i] = std::nextafter(values[i], b[i]);
    }
    return ret;
  }
```

参数表：

- input：输入的第一个Tensor。必须为 float 或 double。
- other：输入的第二个Tensor。数据类型必须与input相同。
- out：输出的Tensor,数据类型必须与input相同。

### 使用示例
```python
>>> import torch
>>> res = torch.nextafter(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0]))
>>> res
tensor([1.0000, 2.0000])
>>> res.backward()
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_3194/3657552041.py in <module>
----> 1 res.backward()

/opt/conda/lib/python3.7/site-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    361                 create_graph=create_graph,
    362                 inputs=inputs)
--> 363         torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
    364 
    365     def register_hook(self, hook):

/opt/conda/lib/python3.7/site-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    173     Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    174         tensors, grad_tensors_, retain_graph, create_graph, inputs,
--> 175         allow_unreachable=True, accumulate_grad=True)  # Calls into the C++ engine to run the backward pass
    176 
    177 def grad(

RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

通过实践验证，我们发现PyTorch中的nextafter不能反向传播梯度。
## 3.2 TensorFlow:
TensorFlow 中有 `tf.math.nextafter` 的API，详细参数为 `tf.math.nextafter(x1, x2, name=None)`。

在 TensorFlow 中的介绍为：

>This operation returns the same result as the C++ std::nextafter function. It can also return a subnormal number..

在实现方法上，TensorFlow 是通过 std::nextafter 实现的，[代码位置](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/nextafter_op.h#L25-L31)
### 实现方法

```cpp
template <typename T>
struct nextafter_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const T operator()(const T& x1,
                                                           const T& x2) const {
    return std::nextafter(x1, x2);
  }
};
```
参数表：

- x1：输入的第一个Tensor。必须为 float64, float32。
- x2：输入的第二个Tensor。数据类型必须与x1相同。
- out：输出的Tensor,数据类型必须与x1相同。
# 四、对比分析

## 共同点

- PyTorch 和 TensorFlow 都是通过 std::nextafter 实现的实现，使用 Python 调用 C++ API 对应的接口。 Paddle 也可以基于 C++ API 实现。

# 五、设计思路与实现方案

## 命名与参数设计

添加 API

```python
paddle.nextafter(
    x: Tensor,
    y: Tensor,
    name: str=None
)
```

## 底层OP设计

底层增加 nextafter OP。因为需要支持广播机制，所以可以模仿Add、Subtract等基础OP设计。根据Pytorch得到的结果，nextafter不需要反向传播，所以不考虑反向算子的实现。

## API实现方案

通过调研发现，需要
1. `paddle/phi/api/yaml/op_compat.yaml`、`paddle/phi/api/yaml/ops.yaml` 添加算子 Nextafter。
2. `paddle/phi/infermeta/binary.cc`、`paddle/phi/infermeta/binary.h` 添加算子 NextafterInferMeta。
3. `paddle/phi/kernels/cpu/` 目录下添加 `nextafter_kernel.cc`文件。
4. `paddle/phi/kernels/gpu/` 目录下添加 `nextafter_kernel.cu`文件。
5. `paddle/phi/kernels/impl/` 目录下添加 `nextafter_kernel_impl.h`文件, C++实现代码。
6. `paddle/phi/kernels/`目录下添加 `nextafter_kernel.h`文件。
7. `python/paddle/__init__.py` 添加 nextafter API，以支持 Tensor.nextafter 的调用方式。
8. `python/paddle/tensor/math.py` 添加Python 实现代码 & 英文 API 文档。
9. `python/paddle/fluid/tests/unittests/` 目录下添加单测文件 `test_nextafter_op.py`。


# 六、测试和验收的考量

测试需要考虑的 case 如下：

- 输出数值结果的一致性和数据类型是否正确，使用 numpy 的 nextafter 函数和新增API进行对比。
- 参数 `x` 的数据类型准确性判断
- 参数 `y` 的数据类型准确性判断

# 七、可行性分析和排期规划

方案主要依赖现有 Paddle 而成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

新增 API，对其他模块无影响。

# 名词解释

无

# 附件及参考资料

[torch.nextafter](https://pytorch.org/docs/2.0/generated/torch.nextafter.html?highlight=nextafter#torch.nextafter)

[Tensor 的广播机制](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/tensor_cn.html#id7)
