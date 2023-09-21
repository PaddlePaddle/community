# paddle.put_along_axis API 增强设计文档

| API名称      | paddle.put_along_axis                     |
| ------------ | -------------------------------------- |
| 提交作者     | mhy                                    |
| 提交时间     | 2023-09-18                             |
| 版本号       | V1.0                                   |
| 依赖飞桨版本  |  develop                                |
| 文件名       | 20230918_api_design_for_put_along_axis.md |


 # 一、概述
 ## 1、相关背景

 当前 paddle.put_along_axis API提供了根据index信息和归约方式累计到原Tensor的对应位置上，希望在此基础上，进一步增强该API的归约功能。

 ## 2、功能目标

 覆盖更全面的归约方式与功能，对应 PyTorch 的 scatter_reduce 操作。

 ## 3、意义

 在图神经网络中，max、min、mean 归约方式也比较常用，增强 put_along_axis API 能扩展用户使用场景。

 # 二、飞桨现状
 当前 paddle 的 put_along_axis 的实现中，支持 assign、add、mul 三种归约方式。

 当前 paddle 的 put_along_axis 的实现是通过根据输入的 index 和 dim 参数计算对应的作用在输出张量的 index，并根据归约方式是 assign、add、mul 分别进行赋值、累加、累乘操作.

 # 三、业内方案调研

 ## PyTorch

 PyTorch 的归约方式支持 sum、prod、mean、amax 和 amin 五种，其中 sum 和 prod 对应 paddle 的 add 和 mul 归约方式，因此需要为 paddle 补充 mean、amin、amax 三种归约方式。

 `Tensor.scatter_reduce_(dim, index, src, reduce, *, include_self=True) → Tensor`

- dim (int) – the axis along which to index
- index (LongTensor) – the indices of elements to scatter and reduce.
- src (Tensor) – the source elements to scatter and reduce
- reduce (str) – the reduction operation to apply for non-unique indices ("sum", "prod", "mean", "amax", "amin")
- include_self (bool) – whether elements from the self tensor are included in the reduction

索引计算方式与 paddle 一致。
 ```python
self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] += src[i][j][k]  # if dim == 2
 ```

paddle不支持include_self=True，默认是 include_self=False的。

paddle不支持 mean/max/min 规约，多了assign规约。

 ### 实现方法

 因为整个算子的实现部分可以分为 index 的计算和归约算子的实现两个部分，而本次任务仅需要增强归约方式，所以下面仅阐述归约算子的实现方法。
 在实现方法上，PyTorch 在 CPU 端的实现如下：

 ``` c++
 class ReduceMultiply {
 public:
   template <typename scalar_t>
   constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
     using opmath_t = at::opmath_type<scalar_t>;
     *self_data *= opmath_t(*src_data);
   }
 };
 static ReduceMultiply reduce_multiply;

 class ReduceAdd {
 public:
   template <typename scalar_t>
   constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
     using opmath_t = at::opmath_type<scalar_t>;
     *self_data += opmath_t(*src_data);
   }
 };
 static ReduceAdd reduce_add;

 class ReduceMean {
 public:
   template <typename scalar_t>
   constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
     using opmath_t = at::opmath_type<scalar_t>;
     *self_data += opmath_t(*src_data);
   }
 };
 static ReduceMean reduce_mean;

 class ReduceMaximum {
 public:
   template <typename scalar_t>
   constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
     using opmath_t = at::opmath_type<scalar_t>;
     *self_data = at::_isnan<scalar_t>(*src_data) ? opmath_t(*src_data) : std::max(*self_data, opmath_t(*src_data));
   }
 };
 static ReduceMaximum reduce_maximum;

 class ReduceMinimum {
 public:
   template <typename scalar_t>
   constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
     using opmath_t = at::opmath_type<scalar_t>;
     *self_data = at::_isnan<scalar_t>(*src_data) ? opmath_t(*src_data) : std::min(*self_data, opmath_t(*src_data));
   }
 };
 static ReduceMinimum reduce_minimum;

 class TensorAssign {
 public:
   template <typename scalar_t>
   constexpr void operator() (at::opmath_type<scalar_t> * self_data, scalar_t * src_data) const {
     using opmath_t = at::opmath_type<scalar_t>;
     *self_data = opmath_t(*src_data);
   }
 };
 static TensorAssign tensor_assign;

// mean
if (op == SCATTER_GATHER_OP::REDUCE_MEAN) {
  auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
  counts.index_add_(dim, index, at::ones_like(source));
  counts.masked_fill_(counts == 0, 1);
  result.div_(counts);
}
 ```

 ## TensorFlow

 目前 TensorFlow 的归约方式支持 add、max、min、sub、update 五种，对应 `tensor_scatter_nd_add`、`tensor_scatter_nd_max`、`tensor_scatter_nd_min`、`tensor_scatter_nd_sub`、`tensor_scatter_nd_update` API，其实现方式也是先求出对应的 index，再进行归约。

 ``` CPP
 // TensorFlow CPU实现
 template <typename T, typename Index>
 struct ScatterFunctorBase<CPUDevice, T, Index, scatter_op::UpdateOp::ASSIGN> {
   Index operator()(OpKernelContext* c, const CPUDevice& d,
                    typename TTypes<T>::Matrix params,
                    typename TTypes<T>::ConstMatrix updates,
                    typename TTypes<Index>::ConstFlat indices) {
     // indices and params sizes were validated in DoCompute().
     const Index N = static_cast<Index>(indices.size());
     const Index limit = static_cast<Index>(params.dimension(0));
     if (!std::is_same<T, tstring>::value) {
       for (Index i = 0; i < N; i++) {
         // Grab the index and check its validity.  Do this carefully,
         // to avoid checking the value and grabbing it again from
         // memory a second time (a security risk since it may change in
         // between).
         const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
         if (!FastBoundsCheck(index, limit)) return i;
         memmove(params.data() + index * params.dimension(1),
                 updates.data() + i * updates.dimension(1),
                 updates.dimension(1) * sizeof(T));
       }
     } else {
       for (Index i = 0; i < N; i++) {
         // Grab the index and check its validity.  Do this carefully,
         // to avoid checking the value and grabbing it again from
         // memory a second time (a security risk since it may change in
         // between).
         const Index index = ::tensorflow::internal::SubtleMustCopy(indices(i));
         if (!FastBoundsCheck(index, limit)) return i;
         // Copy last Ndim-1 dimensions of updates[i] to params[index]
         scatter_op::internal::Assign<scatter_op::UpdateOp::ASSIGN>::Run(
             params.template chip<0>(index), updates.template chip<0>(i));
       }
     }
     return -1;
   }
 };
 ```

 归约的实现：
 ``` c++
 template <>
 struct Assign<scatter_op::UpdateOp::ADD> {
   template <typename Params, typename Update>
   static void Run(Params p, Update u) {
     p += u;
   }
   template <typename Params, typename Update>
   static void RunScalar(Params p, Update u) {
     p = p + u;
   }
 };
 template <>
 struct Assign<scatter_op::UpdateOp::MUL> {
   template <typename Params, typename Update>
   static void Run(Params p, Update u) {
     p *= u;
   }
   template <typename Params, typename Update>
   static void RunScalar(Params p, Update u) {
     p = p * u;
   }
 };
 template <>
 struct Assign<scatter_op::UpdateOp::MAX> {
   template <typename Params, typename Update>
   static void Run(Params p, Update u) {
     p = p.cwiseMax(u);
   }
   template <typename Params, typename Update>
   static void RunScalar(Params p, Update u) {
     p = p.cwiseMax(u);
   }
 };
 ```

 # 四、对比分析

 PyTorch 和 TensorFlow 实现的主要差异在于 index 的计算上，而 paddle 的 put_along_axis 的索引规则与 PyTorch 的 scatter_reduce 以及 TensorFlow 的 scatter_nd 系列不一样并且本次任务并不需要修改索引规则，所以只需要关注归约算子的实现。
 - PyTorch 和 TensorFlow 的归约算子实现原理相同
 - paddle 之前的归约算子的实现与 PyTorch 风格接近且
 - PyTorch 实现了 mean 归约算子而 TensorFlow 没有

 # 五、设计思路与实现方案

 ## 命名与参数设计

 `paddle.put_along_axis(arr, indices, values, axis, reduce='add', include_self=True)`
 `paddle.put_along_axis_(arr, indices, values, axis, reduce='add', include_self=True)`

 其中 put_along_axis_ 是 put_along_axis 的 inplace 版本。

 - `arr (Tensor)` - 输入的 Tensor 作为目标矩阵，数据类型为：float32、float64, int32, int64。 GPU 额外支持float16和bfloat16。
 - `indices (Tensor)` - 索引矩阵，包含沿轴提取 1d 切片的下标，必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐，数据类型为：int、int64。
 - `value （float）` - 需要插入的值，形状和维度需要能够被 broadcast 与 indices 矩阵匹配，数据类型同 arr。
 - `axis (int) - 指定沿着哪个维度获取对应的值，数据类型为：int。`
 - `reduce (str，可选) - 归约操作类型，默认为 add，可选为 add， mul/multiply，amax, amin, mean。不同的规约操作插入值 value 对于输入矩阵 arr 会有不同的行为，如为 assgin 则覆盖输入矩阵，add 则累加至输入矩阵，mul/multiply 则累乘至输入矩阵，amax 则取最大至输入矩阵， amin 则取最小至输入矩阵， mean 则取平均至输入矩阵。`
 - `include_self (bool，可选)` - arr 张量中的元素是否包含在规约中。默认值 include_self = True.


相比于 paddle.put_along_axis 主要差异点为：

1. reduce 目前支持 add、assign、mul/multiply。 reduce=assign在GPU下会出现多种结果，所以此处去除assign的实现，完全和torch.scatter_reduce保持一致。
2. 模型支持 include_self=True的实现，不支持False的实现，且没有对应的参数。
3. 反向梯度计算也存在差异。

 ## 底层OP设计

 在 CPU 端的归约算子实现：

 ``` CPP
 class ReduceMaximum {
  public:
   template <typename tensor_t>
   void operator()(tensor_t* self_data, tensor_t* src_data) const {
     *self_data = std::isnan<tensor_t>(*src_data) ? *src_data : std::max(*self_data, *src_data);
   }
 };
 static ReduceMaximum reduce_maximum;

 class ReduceMinimum {
  public:
   template <typename tensor_t>
   void operator()(tensor_t* self_data, tensor_t* src_data) const {
     *self_data = std::isnan<tensor_t>(*src_data) ? *src_data :  std::min(*self_data, *src_data);
   }
 };
 static ReduceMinimum reduce_minimum;

 class ReduceMean {
  public:
   template <typename tensor_t>
   void operator()(tensor_t* self_data, tensor_t* src_data) const {
     *self_data += *src_data;
   }
 };
 static ReduceMean reduce_mean;
 ```

 mean 归约算子最后在外部除元素个数。
 CUDA 端的归约算子实现：

 ``` CPP
 class ReduceMaximum {
  public:
   template <typename tensor_t>
   __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
     phi::CudaAtomicMax(self_data, *src_data);
   }
 };
 static ReduceMaximum reduce_maximum;

 class ReduceMinimum {
  public:
   template <typename tensor_t>
   __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
     phi::CudaAtomicMin(self_data, *src_data);
   }
 };
 static ReduceMinimum reduce_minimum;

 class ReduceMean {
  public:
   template <typename tensor_t>
   __device__ void operator()(tensor_t* self_data, tensor_t* src_data) const {
     phi::CudaAtomicAdd(self_data, *src_data);
   }
 };
 static ReduceMean reduce_mean;
 ```

 mean 归约算子最后在外部除元素个数。

反向梯度计算逻辑如下：

```c++
  if (reduce == "prod") {
    Tensor masked_self = self.masked_fill(self == 0, 1);
    Tensor masked_self_result = masked_self.index_reduce(dim, index, source, reduce, include_self);
    grad_self = grad * masked_self_result / masked_self;
    Tensor src_zero = source == 0;
    Tensor src_num_zeros = zeros_like(self).index_add(dim, index, src_zero.to(self.dtype())).index_select(dim, index);
    Tensor src_single_zero = bitwise_and(src_zero, src_num_zeros == 1);
    // For src positions with src_single_zero, (grad * result).index_select(dim,index) / source.masked_fill(src_zero, 1)
    // would incorrectly propagate zeros as the gradient
    Tensor masked_src = source.masked_fill(src_single_zero, 1);
    Tensor masked_src_result = self.index_reduce(dim, index, masked_src, reduce, include_self);
    Tensor grad_src1 = where(src_single_zero,
                             (grad * masked_src_result).index_select(dim, index),
                             (grad * result).index_select(dim, index) / source.masked_fill(src_zero, 1));
    if ((src_num_zeros > 1).any().item<bool>()) {
      auto node = std::make_shared<DelayedError>(
        "index_reduce(): Double backward is unsupported for source when >1 zeros in source are scattered to the same position in self",
        /* num inputs */ 1);
      auto result = node->apply({ grad_src1 });
      grad_src = result[0];
    } else {
      grad_src = grad_src1;
    }
  } else if (reduce == "mean") {
    Tensor N = include_self ? ones_like(grad) : zeros_like(grad);
    N = N.index_add(dim, index, ones_like(source));
    N.masked_fill_(N == 0, 1);
    grad_self = grad / N;
    Tensor N_src = N.index_select(dim, index);
    grad_src = grad.index_select(dim, index) / N_src;
  } else if (reduce == "amax" || reduce == "amin") {
    Tensor value = result.index_select(dim, index);
    Tensor self_is_result = (self == result).to(self.scalar_type());
    Tensor source_is_result = (source == value).to(self.scalar_type());
    Tensor N_to_distribute = self_is_result.index_add(dim, index, source_is_result);
    Tensor grad_distributed = grad / N_to_distribute;
    grad_self = self_is_result * grad_distributed;
    grad_src = source_is_result * grad_distributed.index_select(dim, index);
  } else {
    AT_ERROR("Expected 'reduce' to be one of 'prod', 'amax', 'amin' or 'mean' but got ", reduce, ".");
  }

  if (!include_self) {
    grad_self = grad_self.index_fill(dim, index, 0);
  }
```

reduce=sum 的计算逻辑和 mean 类似。

 ## API实现方案

在 python\paddle\tensor\manipulation.py 中修改下 api 和 docstring 即可。
在底层 paddle/phi/kernels/cpu/put_along_axis_kernel.cc 和 paddle/phi/kernels/gpu/put_along_axis_kernel.cu 增加对应的归约算子。
在底层 paddle/phi/kernels/cpu/put_along_axis_grad_kernel.cc 和 paddle/phi/kernels/gpu/put_along_axis_grad_kernel.cu 增加对应的归约梯度算子。

 # 六、测试和验收的考量

 测试考虑的 case 如下：
 - 增加 reduce 分别为 'amin'、'amax' 和 'mean' 时的单测.
 - 增加 include_self 的单侧。
 - 验证反向梯度是否正确。

 # 七、可行性分析和排期规划

 方案实施难度可控，工期上可以满足在当前版本周期内开发完成。

 # 八、影响面

 为已有 API 的增强，对其他模块没有影响

 # 名词解释

 无

 # 附件及参考资料

 无
