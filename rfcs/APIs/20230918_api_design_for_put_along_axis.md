# paddle.put_along_axis 设计文档

| API名称      | paddle.scatter                     |
| ------------ | -------------------------------------- |
| 提交作者     | mhy                                    |
| 提交时间     | 2023-09-18                             |
| 版本号       | V1.0                                   |
| 依赖飞桨版本  |  develop                                |
| 文件名       | 20230918_api_design_for_scatter.md |


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
 ``` CPP
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
 ```

 CUDA 端实现逻辑类似，不过是使用了设备端函数 `fastAtomicAdd`和`gpuAtomicMin`等函数。

 可以看出 Paddle 的 add、assign、mul 归约方式算子与 PyTorch 的对应实现部分是一致的，而 PyTorch 的 amax、amin 和 mean 归约算子的实现也是非常简单直观， amax 算子就是取二者之间的较大值，amin 算子就是取二者之间的较小值，主要变化点在于 mean 算子的实现方式，mean 归约算子的实现方式是先和 add 算子一样进行累加，最后在外部判断如果是 mean 算子的话就除元素个数：（对应文件路径：aten/src/ATen/native/TensorAdvancedIndexing.cpp）

 ```CPP
     if (op == ReductionType::MEAN) {
       auto counts = include_self ? at::ones_like(result) : at::zeros_like(result);
       counts.index_add_(dim, index, at::ones_like(source));
       counts.masked_fill_(counts == 0, 1);
       if (result.is_floating_point() || result.is_complex()) {
         result.div_(counts);
       } else {
         result.div_(counts, "floor");
       }
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

 归约算子的实现：
 ``` CPP
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
 - paddle 之前的归约算子的实现与 PyTorch 风格接近且 PyTorch 实现了 mean 归约算子而 TensorFlow 没有

 # 五、设计思路与实现方案

 ## 命名与参数设计

 paddle.put_along_axis(arr, indices, values, axis, reduce='assign')
 paddle.put_along_axis_(arr, indices, values, axis, reduce='assign')

 其中 put_along_axis_ 是 put_along_axis 的 inplace 版本。

 - `arr (Tensor) - 输入的 Tensor 作为目标矩阵，数据类型为：float32、float64。`
 - `indices (Tensor) - 索引矩阵，包含沿轴提取 1d 切片的下标，必须和 arr 矩阵有相同的维度，需要能够 broadcast 与 arr 矩阵对齐，数据类型为：int、int64。`
 - `value （float）- 需要插入的值，形状和维度需要能够被 broadcast 与 indices 矩阵匹配，数据类型为：float32、float64。`
 - `axis (int) - 指定沿着哪个维度获取对应的值，数据类型为：int。`
 - `reduce (str，可选) - 归约操作类型，默认为 assign，可选为 add， mul，max, min, mean。不同的规约操作插入值 value 对于输入矩阵 arr 会有不同的行为，如为 assgin 则覆盖输入矩阵，add 则累加至输入矩阵，mul 则累乘至输入矩阵，max 则取最大至输入矩阵， min 则取最小至输入矩阵， mean 则取平均至输入矩阵。`


相比于 torch.scatter_reduce 主要差异点为：

1. reduce 新增 max, min, mean 等规约方式。 torch 不支持 assgin。
2. torch 支持 include_self 配置， paddle 使用 include_self=False，此处保持不变。

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


 ## API实现方案

 在底层 paddle\phi\kernels\funcs\gather_scatter_functor.cc 增加对应的归约算子，在 python\paddle\tensor\manipulation.py 中修改下 docstring 即可

 # 六、测试和验收的考量

 测试考虑的 case 如下：
 - 增加 reduce 分别为 'min'、'max' 和 'mean' 时的单测.
 - 验证反向梯度是否正确。

 # 七、可行性分析和排期规划

 方案实施难度可控，工期上可以满足在当前版本周期内开发完成。

 # 八、影响面

 为已有 API 的增强，对其他模块没有影响

 # 名词解释

 无

 # 附件及参考资料

 无
