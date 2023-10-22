# paddle.masked_scatter 设计文档

|API名称 | paddle.masked_scatter                     |
|---|-------------------------------------------|
|提交作者| yangguohao                                |
|提交时间 | 2023-10-22                                |
|版本号 | V1.0                                      |
|依赖飞桨版本| develop                                   |
|文件名 | 20231022_api_design_for_masked_scatter.md |


# 一、概述
## 1、相关背景

飞桨黑客松第五期 No.4：为 Paddle 新增 masked_scatter API。 `masked_scatter` 根据mask信息，将value中的值逐个拷贝到原Tensor的对应位置上。


## 2、功能目标

此任务的目标是在 Paddle 框架中，新增 masked_scatter API ，调用路径为：

* paddle.masked_scatter 作为独立的函数调用，非 inplace
* paddle.masked_scatter_，作为独立的函数，inplace 地修改输入；
* Tensor.masked_scatter作为 Tensor 的方法使用，非 inplace;
* Tensor.masked_scatter_作为 Tensor 的方法使用， inplace 修改输入；

## 3、意义

该功能被广泛使用到，因此在Paddle中提供该API，方便用户使用。且对标 pytorch 中已有的 api，同时也可以方便实现代码转换。

# 二、飞桨现状

目前paddle缺少相关功能实现。该 RFC 考虑通过 CPU 和 GPU kernel 来实现该算子 OP

# 三、业内方案调研

## PyTorch

PyTorch中提供了`Tensor.masked_scatter(mask, tensor)`以及`Tensor.masked_scatter_(mask, source)`两个API，

GPU 的实现如下
```cpp

void launch_masked_scatter_kernel(
    const TensorBase &self, const TensorBase &mask,
    const TensorBase &maskPrefixSum, const TensorBase &source) {
  const auto srcSize = source.numel();
  const auto mask_cont = mask.contiguous();
  const auto mask_numel = mask.numel();

  // Use a prefix sum to determine the output locations of the masked elements
  auto maskPrefixSum_data = maskPrefixSum.mutable_data_ptr<int64_t>();
  auto mask_data = mask_cont.const_data_ptr<bool>();

  at::cuda::cub::mask_exclusive_sum(
      mask_data, maskPrefixSum_data, mask_numel);

  // Asynchronously check that the number of `1` elements present in the mask
  // must be <= the number of elements available in `src`.
  masked_scatter_size_check<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      &maskPrefixSum_data[mask_numel - 1], &mask_data[mask_numel - 1], srcSize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  auto source_contig = source.contiguous();

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(self)
      .add_input(mask_cont)
      .add_input(maskPrefixSum)
      .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      self.scalar_type(),
      "masked_scatter_",
      [&]() {
        auto source_ptr = source_contig.const_data_ptr<scalar_t>();
        gpu_kernel(
            iter, [=] GPU_LAMBDA(const scalar_t a, const bool mask, const int64_t maskPrefixSum) -> scalar_t {
              if (mask) {
                return source_ptr[maskPrefixSum];
              }
              return a;
            });
        AT_CUDA_CHECK(cudaGetLastError());
      });
}
```

对于GPU实现，PyTorch的主要的过程即首先计算 mask_prefix_sum 然后通过以下核心逻辑来进行

```cpp
// 计算maskPrefixSum
// ...
if (mask) {
    return source_ptr[maskPrefixSum];
}
return a;
```

CPU 代码实现如下
```cpp
template <typename scalar_t>
void cpu_masked_scatter_kernel(TensorIterator& iter, const TensorBase& source) {
  std::ptrdiff_t source_cntr = 0;
  scalar_t* source_ptr = source.data_ptr<scalar_t>();
  auto numel = source.numel();
  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    const int64_t dst_stride = strides[0];
    char* mask = data[1];
    const int64_t mask_stride = strides[1];
    for (const auto i : c10::irange(n)) {
      auto mask_value = *reinterpret_cast<bool*>(mask + mask_stride * i);
      if (mask_value) {
        TORCH_CHECK(source_cntr < numel, "Number of elements of source < number of ones in mask");
        *(scalar_t*)(dst + dst_stride * i) = *(source_ptr);
        source_ptr++;
        source_cntr++;
      }
    }
  };
  iter.serial_for_each(loop, {0, iter.numel()});
}
void masked_scatter_kernel(TensorIterator& iter, const TensorBase& source) {
 TORCH_CHECK(iter.input_dtype() == ScalarType::Bool, "masked_scatter_ only supports boolean masks, "
    "but got mask with dtype ", iter.input_dtype());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.dtype(),
      "masked_scatter",
      [&] {
          cpu_masked_scatter_kernel<scalar_t>(iter, source);
      });
}
```
对于CPU实现，PyTorch的实现思路类似于用一个 for 循环函数来进行上面核心逻辑的计算。

2. 反向部分

Pytorch 的 derivatives.yaml 中对于 self 的 grad 用了 masked_fill
```
- name: masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
  self: grad.masked_fill(mask, 0)
  source: masked_scatter_backward(grad, mask, source.sym_sizes())
  mask: non_differentiable
  result: self_t.masked_scatter(mask, source_t)
```
对于 source 的 grad，则通过 masked_select 加上 fill_zero 的方法完成。
```cpp
Tensor masked_scatter_backward(
    const Tensor& grad,
    const Tensor& mask,
    c10::SymIntArrayRef sizes) {
  c10::SymInt numel = 1;
  for (const auto& size : sizes) {
    numel *= size;
  }
  auto mask_selected = grad.masked_select(mask);
  auto diff_nelem = numel - mask_selected.sym_numel();
  if (diff_nelem > 0) {
    // because mask_selected returns a 1-d tensor with size of masked elements
    // that are 1, we need to fill out the rest with zeros then reshape back to
    // tensor2's size.
    auto zeros_fillin =
        at::zeros_symint({std::move(diff_nelem)}, grad.options());
    mask_selected = at::cat({mask_selected, std::move(zeros_fillin)}, 0);
  }
  return mask_selected.view_symint(sizes);
}

```
## TensorFlow

TensorFlow中没有`masked_scatter`API的实现

# 四、对比分析

- Pytorch 自定义Kernel的方式更加高效

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.masked_scatter(x, mask, source, name=None)

paddle.masked_scatter_(x, mask, source, name=None)

Tensor.masked_scatter(mask, source, name=None)

Tensor.masked_scatter_(mask, source, name=None)
```
masked_scatter和masked_scatter_分别表示out-place和in-place两种计算形式。

- `x (Tensor)`: 输入的张量，支持的数据类型为float16、float32、float64、int32、int64。
- `mask (Tensor, bool)`: 用于指定填充位置的布尔值掩码张量，与 input 的 shape 相同，或者可以广播成 x 的 shape。
- `value (Tensor)`: 待填充的张量，支持的数据类型为float16、float32、float64、int32、int64，其中元素的数量应该不少于mask中True的个数，且元素数据类型要跟x中元素数据类型保持一致。
- `name (str，可选)` :一般无需设置，默认值为 None。

## 底层OP设计

参考 pytorch 中的设计：
1. 前向部分:
   - CPU 部分主要通过 for 循环依次根据mask信息，将value中的值逐个拷贝到原Tensor的对应位置上。
   - GPU 同样通过 cast + cumsum 先计算 mask_prefix_sum 再通过 ElementWiseKernel 将对应的值拷贝到原 Tensor 的对应位置中.
2. 反向部分:
   - CPU 部分主要通过 for 循环分别处理 x_grad 和 source_grad。
   - GPU 部分可以使用 ElementWiseKernel 计算 x_grad，对于 source_grad 同样用 masked_select + fill_zeros，concat 拼接后 resize 得到。

该部分的代码已经编写测试，待 RFC 通过后就可以提 PR。



## API实现方案

```python
def masked_scatter(x, mask, source):
    if in_dynamic_mode():
        return _C_ops.masked_scatter(x, mask, source)
    else:
        check_variable_and_dtype(mask, 'mask', ['bool'], 'masked_scatter')
        check_variable_and_dtype(
            x,
            'x',
            ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'masked_scatter',
        )
        check_variable_and_dtype(
            source,
            'source',
            ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'masked_scatter',
        )
        helper = LayerHelper("masked_scatter", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='masked_scatter',
            inputs={
                'x': x,
                'mask': mask,
                'source': source,
            },
            outputs={'out': [out]},
        )

        return out
```
# 六、测试和验收的考量

1. 添加单测文件 `paddle/test/legacy_test/test_masked_scatter_op.py`。
2. 同时在 `paddle/test/legacy_test/test_inplace.py` 中新增对应的inplace api 单测


测试需要考虑的 case 如下：

- 输入的mask和input的形状不一致，但是可以broadcast
- 检查算子计算结果的正确性，以pytorch为参考
- 测试在进行反向梯度计算时结果的正确性
- 错误检查：输入x不满足要求时,能否正确抛出错误
- 错误检查：需要考虑当mask的true元素数量大于 source 的数量时报错。
- 错误检查：mask 类型判断为 bool。

# 七、可行性分析和排期规划

方案主要利用paddle现有api完成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面
新增 OP 以及 API，对其他模块无影响。

# 名词解释

# 附件及参考资料
[paddle.index_put_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/index_put__cn.html#cn-api-paddle-index-put)

[TORCH.TENSOR.MASKED_SCATTER_](https://pytorch.org/docs/2.0/generated/torch.Tensor.masked_scatter_.html#torch.Tensor.masked_scatter_)
