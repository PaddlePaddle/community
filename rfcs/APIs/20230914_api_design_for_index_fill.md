# paddle.index_fill 设计文档

| API名称                                                      | paddle.index_fill |
| ------------------------------------------------------------ | ----------------------------------- |
| 提交作者   | NetPunk                   |
| 提交时间| 2023-09-14                 |
| 版本号                                                       | V1.0                                |
| 依赖飞桨版本 | develop                             |
| 文件名                                                       | 20220316_api_design_for_index_fill.md |

# 一、概述

## 1、相关背景

对于 nd tensor, 沿着某个轴 axis 取 (n-1)d 的切片，索引位置是 index, 并且将 value 中值填充到这些切片上。其中 value 是一个 scalar 或者 0d tensor, 该运算需要支持微分。

## 2、功能目标

index_fill API 是一个按轴和索引填充值到目标张量的API。此任务的目标是在 Paddle 框架中，新增 index_fill API，同时实现inplace和非inplace版本，调用路径为：

- paddle.index_fill 作为独立的函数调用，非 inplace
- paddle.index_fill_，作为独立的函数，inplace 地修改输入；
- Tensor.index_fill， 作为 Tensor 的方法使用，非 inplace;
- Tensor.index_fill_，作为 Tensor 的方法使用， inplace 修改输入；

## 3、意义

完善Paddle API丰富度

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch 中有inplace操作 API（https://pytorch.org/docs/stable/generated/torch.Tensor.index_fill_.html）

在 PyTorch 文档中，介绍为：

```
Fills the elements of the self tensor with value value by selecting the indices in the order given in index.

Parameters:
 - dim (int) – dimension along which to index
 - index (LongTensor) – indices of self tensor to fill in
 - value (float) – the value to fill with
```
输入用于定位的dim和index，原地修改tensor对应位置的值为value

### 实现方法

在实现方法上, PyTorch采用的CPU实现为：循环遍历赋值，而CUDA实现则是调用pytorch自己实现的scan_with_indices函数。
核心代码为：
[CPU](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp#L769):

```cpp
void index_fill_kernel(
  TensorIterator& iter,
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride,
  const Scalar& source) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, kComplexHalf,
    iter.dtype(), "index_fill_cpu", [&] {
    auto fill_val = source.to<scalar_t>();
    auto handle_nonzero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {
      auto* self_data_bytes = data[0];
      auto* index_data_bytes = data[1];
      for (const auto elem C10_UNUSED : c10::irange(n)) {
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);
        auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
        TORCH_CHECK_INDEX(idx >= -self_dim_size && idx < self_dim_size,
                          "index ", idx, " is out of bounds for dimension ",
                          dim, " with size ", self_dim_size);
        if (idx < 0) {
          idx += self_dim_size;
        }

        self_data[idx * self_dim_stride] = fill_val;

        self_data_bytes += strides[0];
        index_data_bytes += strides[1];
      }
    };
    auto handle_zero_idx_stride = [&](char** data, const int64_t* strides, int64_t n) {
      auto* self_data_bytes = data[0];
      auto* index_data_bytes = data[1];
      auto idx = *reinterpret_cast<int64_t*>(index_data_bytes);
      TORCH_CHECK_INDEX(idx >= -self_dim_size && idx < self_dim_size,
                        "index ", idx, " is out of bounds for dimension ",
                        dim, " with size ", self_dim_size);
      if (idx < 0) {
        idx += self_dim_size;
      }
      for (const auto elem C10_UNUSED: c10::irange(n)) {
        auto* self_data = reinterpret_cast<scalar_t*>(self_data_bytes);

        self_data[idx * self_dim_stride] = fill_val;

        self_data_bytes += strides[0];
      }
    };

    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
      auto idx_stride = strides[1];
      if (idx_stride) {
        handle_nonzero_idx_stride(data, strides, n);
      }
      else {
        handle_zero_idx_stride(data, strides, n);
      }
    };
    iter.for_each(loop);
  });
}
```
[GPU](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/ScanKernels.cpp#L28):

```cpp
template <typename scalar_t>
void index_fill_kernel_impl(
  TensorIterator& iter,
  int64_t dim,
  int64_t self_dim_size,
  int64_t self_dim_stride,
  scalar_t fill_val) {
  if (0 == iter.numel()) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      index_fill_kernel_impl(sub_iter, dim, self_dim_size, self_dim_stride, fill_val);
    }
    return;
  }

  char* __restrict__ self_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* __restrict__ idx_ptr = reinterpret_cast<char*>(iter.data_ptr(1));

  auto offset_calc = make_offset_calculator<2>(iter);

  auto loop = [=]C10_DEVICE(int i) {
    auto offsets = offset_calc.get(i);

    auto* __restrict__ self_data = reinterpret_cast<scalar_t*>(self_ptr + offsets[0]);
    auto idx = *reinterpret_cast<int64_t*>(idx_ptr + offsets[1]);
    CUDA_KERNEL_ASSERT(idx >= -self_dim_size && idx < self_dim_size && "index out of bounds");
    if (idx < 0) {
      idx += self_dim_size;
    }

    self_data[idx * self_dim_stride] = fill_val;
  };
  launch_kernel<launch_size_nd, launch_bound2>(iter.numel(), loop);
}
```
可以看出实现思路和计算思路是比较一致的，先将tensor展开，再以指针跳跃扫描的方式赋值，最后还原形状



## Paddle

Paddle已经实现了index_put API用于依据索引 `indices` ，将指定位置的 `x` 重新赋值为 `value`，链接：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/index_put__cn.html

其API的用法和index_fill不甚相同，但是功能上可以覆盖，可以说更加灵活



# 四、对比分析

可以直接参考的实现是pytorch，但是鉴于paddle中已有index_put API，可以想到组合index_put和其它Paddle API，在python端实现index_fill的功能，由此利用index_put已经实现的动静图、前反向功能



# 五、方案设计

## 命名与参数设计

API设计为`paddle.index_fill(x, axis, index, value, name)`以及`paddle.index_fill_(x, axis, index, value, name)`。

paddle.index_fill
----------------------
参数
:::::::::

- x (Tensor) - 需要填充的目标张量，支持类型int32, int64, float32, float64。
- axis (int) - 做索引操作的维度。-1代表最后一维。默认：None，将输入展开为一维变量再进行累积最大值计算。
- index (Tensor) - 包含索引的一维张量，可以为int32和int64
- value (float) - 张量填充的值，可以为int32, int64, float32, float64
- name  (str) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。

:::::::::

- out(Tensor) - 返回填充后的张量。数据类型和输入`x`一致。

paddle.Tensor.index_fill指向paddle.index_fill，两者是相同的API

paddle.index_fill_
----------------------

参数
:::::::::

- x (Tensor) - 需要填充的目标张量，支持类型int32, int64, float32, float64。
- axis (int) - 做索引操作的维度。-1代表最后一维。默认：None，将输入展开为一维变量再进行累积最大值计算。
- index (Tensor) - 包含索引的一维张量，可以为int32和int64
- value (float) - 张量填充的值，可以为int32, int64, float32, float64
- name  (str) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。

paddle.Tensor.index_fill_指向paddle.index_fill\_，两者是相同的API

## 底层OP设计

python端API组合实现

## API实现方案

参考 PyTorch 的计算逻辑，先将输入Tensor展开，再构造index_put的输入参数，最后返回形状复原后的结果，初版代码如下

~~~python
def compute_stride(axis, dims):
    size = 1
    for i in range(axis + 1, len(dims)):
        size *= dims[i]
    return size

if isinstance(index, paddle.Tensor):
    index = index.numpy()

ndims = len(x.shape)
finished = 0
counter = [0] * ndims
x_data = 0
x_stride = compute_stride(axis, x.shape)
x_dim_vec = x.shape
out = paddle.to_tensor(x)
out = paddle.flatten(out)
idx = []

while finished == 0:
    for i in index:
        idx.append(x_data + i * x_stride)
    if ndims == 1: break
    for dim_i in range(ndims):
        if dim_i == axis:
            if dim_i == ndims - 1:
                finished = 1
                break
            continue
        x_stride_ = compute_stride(dim_i, x_dim_vec)
        counter[dim_i] += 1
        x_data += x_stride_
        if counter[dim_i] == x_dim_vec[dim_i]:
            if dim_i == ndims - 1:
                finished = 1
                break
            else:
                x_data -= counter[dim_i] * x_stride_
                counter[dim_i] = 0
        else:
            break

values = paddle.to_tensor([value] * len(idx))
idx = paddle.to_tensor(idx)
indices = (idx,)
out = paddle.index_put(out, indices, values, accumulate=False)
return paddle.reshape(out, x_dim_vec)
~~~

索引的遍历参考了cummax/cummin算子的CPU实现，[链接](https://github.com/PaddlePaddle/Paddle/pull/53546/files#diff-0417a927e0148c22ecb722f950e2f9704d6e899e9899521f0a269b173ceb2de2)



# 六、测试和验收的考量

测试考虑的case如下：

- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算和反向计算；
  - 计算dtype类型：验证 `float64`，`int32`等；
  
- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：输入参数类型、形状的有效性校验。

# 七、可行性分析及规划排期

技术可行性：参考同类项目和相似的 API，无重大难点；

1st week：中英文 API 文档编写 & 测试样例；

2nd week：前端 Python 代码编写；

3th week：测试和完善文档。

# 八、影响面

为独立新增API，对其他模块没有影响