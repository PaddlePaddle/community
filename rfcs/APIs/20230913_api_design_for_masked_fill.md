# paddle.masked_fill 设计文档

| API名称      | paddle.masked_fill                     |
| ------------ | -------------------------------------- |
| 提交作者     | AndSonder                              |
| 提交时间     | 2023-09-13                             |
| 版本号       | V1.0                                   |
| 依赖飞桨版本 | develop                                |
| 文件名       | 20230913_api_design_for_masked_fill.md |

# 一、概述

## 1、相关背景

`masked_fill` 是一个常用的API，该 API 的作用是根据 `mask` 信息，将 `value` 中的值填充到 `Tensor` 中 `mask` 对应为 `True` 的位置。这个功能在语义分割、序列标注等任务中经常用到。因此，在Paddle中提供该API，方便用户使用。

## 2、功能目标

在 Paddle 框架中，新增 `paddle.masked_fill` 对于一个Tensor，根据mask信息，将 value 中的值填充到该Tensor中mask对应为True的位置。

## 3、意义

该API是一个常用的API，可以方便用户使用。让用户不用自己实现该功能，提高用户的使用效率。

# 二、飞桨现状

目前paddle缺少相关功能实现。只能通过 paddle 现有的 API 组合实现。

```python
# paddlepaddle >= 2.0
import paddle

paddle.seed(123)
x = paddle.ones([3, 3], dtype='float32')
# Tensor(shape=[3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[0.00276479, 0.45899123, 0.96637046],
#         [0.66818708, 0.05855134, 0.33184195],
#         [0.34202638, 0.95503175, 0.33745834]])

mask = paddle.randint(0, 2, [3, 3]).astype('bool')
# Tensor(shape=[3, 3], dtype=bool, place=CUDAPlace(0), stop_gradient=True,
#        [[True , True , False],
#         [True , True , True ],
#         [True , True , True ]])

def masked_fill(x, mask, value):
    y = paddle.full_like(x, value, x.dtype)
    return paddle.where(mask, y, x)

out = masked_fill(x, mask, 2)
# Tensor(shape=[3, 3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [[2.        , 2.        , 0.96637046],
#         [2.        , 2.        , 2.        ],
#         [2.        , 2.        , 2.        ]])
```


full/full_like 和 where 均支持在 CPU 和 GPU 上运行。

paddle.full_like 支持的参数 dtype:

```python 
CPU Kernel 
float,double,int8_t,uint8_t,int16_t,int,int64_t,bool,float16,bfloat16,complex32,complex64

GPU Kernel
float,double,int8_t,uint8_t,int16_t,int,int64_t,bool,float16,bfloat16,complex32,complex64
```

paddle.where 支持的参数 dtype:

```python 
CPU Kernel 
float, double, int, int64_t

GPU Kernel
float,double,int,int64_t,float16,bfloat16
```

使用 full/full_like 和 where 组合完成的 masked_fill API，支持 broadcast 机制。

```python
x = paddle.ones([3, 3], dtype='float32')
mask = paddle.randint(0, 2, [1, 3]).astype('bool')

out = masked_fill(x, mask, 2)
print(out)

# Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[2., 1., 2.],
#         [2., 1., 2.],
#         [2., 1., 2.]])
```



# 三、业内方案调研

## Pytorch

Pytorch中 有 API `Tensor.masked_fill_(mask, value)`

在pytorch中，介绍为：

```
Fills elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.
```

其中输入参数的描述如下：

- mask (BoolTensor) – the boolean mask
- value (float) – the value to fill in with

### 实现方法


在实现方法上, Pytorch 设计了两种实现方式，一种是CPU实现，一种是GPU实现。



核心代码如下：

```cpp
// GPU 实现
void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kBool, kHalf, kBFloat16, kComplexHalf, iter.common_dtype(), "masked_fill_", [&]() {
        const auto value_ = value.to<scalar_t>();
        gpu_kernel(
            iter, [value_] GPU_LAMBDA(scalar_t self, bool mask) -> scalar_t {
              if (mask) {
                return value_;
              }
              return self;
            });
      });
}

// CPU 实现
template <typename scalar_t>
void cpu_masked_fill_kernel(TensorIterator& iter, scalar_t value) {
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    char* mask = data[1];
    for (const auto i : c10::irange(n)) {
      bool mask_value = *reinterpret_cast<bool*>(mask + strides[1] * i);

      if (mask_value) {
        *(scalar_t*)(dst + strides[0] * i) = value;
      }
    }
  };
  iter.for_each(loop);
}

void masked_fill_kernel(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(kComplexHalf, kBool, kBFloat16, kHalf,
    iter.dtype(), "masked_fill", [&] {
      scalar_t scalar_val = value.to<scalar_t>();
      auto mask_dtype = iter.input_dtype(0);
      TORCH_CHECK(mask_dtype == ScalarType::Bool, "masked_fill only supports boolean masks, "
        "but got mask with dtype ", mask_dtype);
      cpu_masked_fill_kernel<scalar_t>(iter, scalar_val);
    });
}

```

Pytorch 在 CPU 和 GPU 上对 masked_fill 的实现方式有些不同:

CPU 实现:

1. 使用模板函数 cpu_masked_fill_kernel 来实现标量值填充逻辑。

2. TensorIterator::for_each 启动循环,对每组数据调用 lambda 函数。

3. lambda 中直接访问指针进行填充判断和赋值。

4. 使用宏生成不同数据类型的特化模板。

5. 调用入口做参数校验。

GPU 实现: 

1. 用 gpu_kernel 启动 CUDA kernel。

2. 用 GPU Lambda 编写 kernel 函数体。

3. kernel 函数签名为 (value, mask) -> output, 直接在 GPU 上判断和赋值。

4. 用宏生成不同数据类型的 kernel。

5. 调用入口转换 value 为模板类型。



## Tensorflow

Tensorflow 并没有直接提供 `masked_fill` 的API，但是可以通过 `tf.where` 来实现。相关讨论PR: https://github.com/tensorflow/tensorflow/pull/41975

讨论结果为使用 `tf.where` 实现 `masked_fill` 的功能更加高效，因此没有提供 `masked_fill` 的API。

# 四、对比分析

- Pytorch 自定义Kernel的方式更加高效
- Tensorflow 通过 `tf.where` 实现 `masked_fill`


# 五、方案设计

## 命名与参数设计

paddle.masked_fill(input, mask, value)

paddle.masked_fill_(input, mask, value)

Tensor.masked_fill(mask, value)

Tensor.masked_fill_(mask, value)

masked_fill_支持inplace方式修改输入张量。

- `input (Tensor)`: 输入的张量，需要进行填充操作。
- `mask (Tensor, bool)`: 用于指定填充位置的布尔值掩码张量，与 input 张量形状相同。
- `value (Tensor, bool, int, float, complex)`: 待填充的数据，参数类型支持布尔值、整数、浮点数以及0维的张量。
- `inplace (bool, optional)`: 是否进行 inplace 操作。如果设置为 True，则会直接修改输入张量，否则返回一个新的张量，默认为 False。


## 底层OP设计

依赖python实现，无需底层op支持。

## API实现方案

在 python/paddle/tensor/manipulation.py 中增加 masked_fill 以及 masked_fill_ 函数。

通过 `paddle.full_like` 和 `paddle.where` 组合实现。

```python
out = paddle.full(x.shape, value, x.dtype)
out = paddle.where(mask, y, x)
```

## 代码实现文件路径

函数API实现路径: python/paddle/tensor/manipulation.py

单元测试路径：在 Paddle repo 的 test/ 目录, 同时在 paddle/test/legacy_test/test_inplace.py 中新增对应的inplace api 单测


# 六、测试和验收的考量

测试考虑的case如下：

- 输入的mask和input的形状不一致，但是可以broadcast
- 校验参数 value 的正确性， 是否是支持的数据类型，当 value 是0维 tensor 时梯度正确回传
- 测试在进行反向梯度计算时结果的正确性
- 错误检查：输入x不是Tensor时,能否正确抛出错误


# 七、可行性分析及规划排期

方案实施难度可控，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无