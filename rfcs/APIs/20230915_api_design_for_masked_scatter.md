# paddle.masked_scatter 设计文档

|API名称 | paddle.masked_scatter | 
|---|---|
|提交作者| ranchongzhi | 
|提交时间 | 2023-09-15 | 
|版本号 | V1.0 | 
|依赖飞桨版本| develop | 
|文件名 | 20230915_api_design_for_masked_scatter.md | 


# 一、概述
## 1、相关背景

`masked_scatter`是一个常用的api，用于根据给定的`mask`以及源`Tensor`在目标`Tensor`指定位置上进行替换操作，这个功能在序列标注等任务中经常被用到，因此，在Paddle中提供该API，方便用户使用。

## 2、功能目标

在 Paddle 框架中，新增 `paddle.masked_scatter` 对于一个`目标Tensor`，根据`mask`信息，将`源Tensor`中的值按照顺序填充到`目标Tensor`中`mask`对应为`True`的位置。

## 3、意义

该API是一个常用的API，可以方便用户使用。让用户不用自己实现该功能，提高用户的使用效率。

# 二、飞桨现状

目前paddle缺少相关功能实现。只能通过 paddle 现有的 API 组合实现。

# 三、业内方案调研

## PyTorch

PyTorch中提供了`Tensor.masked_scatter(mask, tensor)`以及`Tensor.masked_scatter_(mask, source)`两个API，分别表示`out-place`和`in-place`两种计算形式，在pytorch中的介绍如下：

```
Copies elements from source into self tensor at positions where the mask is True. Elements from source are copied into self starting at position 0 of source and continuing in order one-by-one for each occurrence of mask being True. The shape of mask must be broadcastable with the shape of the underlying tensor. The source should have at least as many elements as the number of ones in mask.
```
其中输入参数的描述如下：

- mask (BoolTensor) – the boolean mask
- source (Tensor) – the tensor to copy from
### 实现方法

在实现方法上, Pytorch 设计了两种实现方式，一种是CPU实现，一种是GPU实现，核心代码如下：

```cpp
//GPU实现
//代码路径：pytorch/aten/src/ATen/native/cuda/IndexKernel.cu
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
对于GPU实现，PyTorch的实现思路如下：
1. 首先，获取源张量的元素数量（srcSize）以及掩码张量的连续版本（mask_cont）和元素数量（mask_numel）。
   
2. 使用前缀和（prefix sum）计算掩码元素的输出位置。maskPrefixSum是一个输入张量，用于存储掩码元素的前缀和结果。maskPrefixSum_data和mask_data分别获取了maskPrefixSum和mask_cont的数据指针。
   
3. 异步地检查掩码中为1的元素数量是否小于等于源张量中的元素数量。这是通过调用masked_scatter_size_check内核函数实现的，该函数使用CUDA流来执行。这个检查确保源张量中有足够的元素供复制到掩码为True的位置。
   
4. 将源张量转换为连续版本（source_contig），以便可以根据maskPrefixSum的偏移量从source中获取元素。
   
5. 创建一个TensorIterator对象（iter），配置它的参数，包括指定输出张量（self）和输入张量（self、mask_cont、maskPrefixSum），并构建该迭代器。
   
6. 使用AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3宏对数据类型进行分派，包括除Bool、BFloat16和Half之外的所有类型。这个宏用于在CUDA内核函数中处理不同的标量类型。
7. 在内核函数中，获取源张量的数据指针（source_ptr），然后使用gpu_kernel函数执行操作。该操作是一个Lambda函数，根据掩码值（mask）和对应位置的前缀和（maskPrefixSum）选择要返回的值。如果掩码为True，则返回source_ptr[maskPrefixSum]，否则返回输入的值（a）。
8. 最后，使用cudaGetLastError检查CUDA内核函数是否执行成功


```cpp
//CPU实现
//代码路径：pytorch/aten/src/ATen/native/cpu/IndexKernel.cpp
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
对于CPU实现，PyTorch的实现思路如下：
1. 实现一个名为cpu_masked_scatter_kernel的模板函数，用于处理CPU上的masked_scatter操作。它接受一个TensorIterator对象和一个源张量source作为参数。
2. 在cpu_masked_scatter_kernel函数中，获取源张量的数据指针（source_ptr）和元素数量（numel）。通过调用TensorIterator::serial_for_each启动循环，对每组数据都调用loop函数。
3. loop函数中通过直接访问指针进行填充判断，数据赋值等逻辑。
4. 使用宏AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3生成针对于不同数据的类型的特化模板
5. 调用入口对mask中元素的类型进行校验。

## TensorFlow

TensorFlow中没有`masked_scatter`API的实现

## Scipy

Scipy中没有`masked_scatter`API的实现

## Numpy

Numpy中没有`masked_scatter`API的实现

# 四、对比分析

- Pytorch 自定义Kernel的方式更加高效
- Tensorflow、Scipy、Numpy中没有`masked_scatter`API的实现

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.masked_scatter(input, mask, source)

paddle.masked_scatter_(input, mask, source)

Tensor.masked_scatter(mask, source)

Tensor.masked_scatter_(mask, source)
```
masked_scatter和masked_scatter_分别表示out-place和in-place两种计算形式。

- `input (Tensor, float, double, int, int64_t, float16, bfloat16)`: 输入的张量，需要根据mask进行赋值操作。
- `mask (Tensor, bool)`: 用于指定填充位置的布尔值掩码张量，与 input 张量形状相同，或者可以广播成input张量的形状。
- `source (Tensor, float, double, int, int64_t, float16, bfloat16)`: 待填充的张量，其中元素的数量应该不少于mask中True的个数。
- `name (str，可选)` :一般无需设置，默认值为 None。


## API实现方案

C ++/CUDA 参考 PyTorch 实现，实现位置为 Paddle repo `paddle/phi/kernels` 目录，cc 文件在 `paddle/phi/kernels/cpu` 目录和 cu 文件在 `paddle/phi/kernels/gpu` 目录。

Python 实现代码 & 英文 API 文档，放在 Paddle repo 的 `python/paddle/tensor/manipulation.py` 文件。并在 `python/paddle/tensor/init.py` 中，添加 masked_scatter & masked_scatter_ API，以支持 paddle.Tensor.masked_scatter & paddle.Tensor.masked_scatter_ 的调用方式。

# 六、测试和验收的考量
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)
1. 添加单测文件 `test/legacy_test/test_masked_scatter_op.py`。
2. 在单测文件 `test/legacy_test/test_inplace.py` 补充测试。

测试需要考虑的 case 如下：

- 输出数值结果的一致性和数据类型是否正确，使用 PyTorch 作为参考标准
- 对不同 dtype 的输入数据 `x` 进行计算精度检验 (float32, float64)
- 输入输出的容错性与错误提示信息
- 输出 Dtype 错误或不兼容时抛出异常
- 保证调用属性时是可以被正常找到的
- 覆盖静态图和动态图测试场景

# 七、可行性分析和排期规划

方案主要参考 PyTorch 的工程实现方法，工期上可以满足在当前版本周期内开发完成。

# 八、影响面
新增 API，对其他模块无影响

# 名词解释

# 附件及参考资料
[TORCH.TENSOR.MASKED_SCATTER_](https://pytorch.org/docs/2.0/generated/torch.Tensor.masked_scatter_.html#torch.Tensor.masked_scatter_)