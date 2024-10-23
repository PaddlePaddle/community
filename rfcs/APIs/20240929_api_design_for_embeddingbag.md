# paddle.nn.functional.embedding_bag \ paddle.nn.EmbeddingBag 设计文档

| API 名称     | paddle.nn.functional.embedding_bag \ paddle.nn.EmbeddingBag |
| ------------ | ---------------------------------------- |
| 提交作者     | NKNaN                                      |
| 提交时间     | 2024-09-29                               |
| 版本号       | V1.0                                     |
| 依赖飞桨版本 | develop                                  |
| 文件名       | 20240929_api_design_for_embedding_bag.md |

# 一、概述

## 1、相关背景
[NO.24 为 Paddle 新增 EmbeddingBag API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/【Hackathon%207th】个人挑战赛—框架开发任务合集.md#no24-为-paddle-新增-embeddingbag-api)
`EmbeddingBag` 是 `Embedding` 的拓展，此任务的目标是在 Paddle 框架中，新增 EmbeddingBag 和 embedding_bag API，调用路径为：`paddle.nn.EmbeddingBag` 和 `paddle.nn.functional.embedding_bag`。

## 2、功能目标
EmbeddingBag 算子是在不实例中间变量的情况下实现求和/求均值等系列运算，因此其算子的构建目标为：相比组合使用 Embedding 和 reduce 方法（sum/mean/max），单独使用 EmbeddingBag 能够提高计算效率并且减少内存消耗。

## 3、意义
提高 Embedding 和 reduce 方法（sum/mean/max）组合时的计算效率，丰富 paddle API。

# 二、飞桨现状
目前 paddle 缺少相关功能实现

根据功能目标不可以使用组合方案替代实现

# 三、业内方案调研
Pytorch 和 Tensorflow 均有 embedding_bag 实现，调研如下：

## PyTorch
CPU设备
```cpp
template <typename scalar_t>
void embedding_bag_cpu_max_out(
    Tensor* max_indices,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& output,
    bool include_last_offset,
    Tensor& bag_size,
    int64_t padding_idx) {
  int64_t numIndices = indices.numel();
  int64_t featureSize = weight.size(1);
  int64_t vocab_size = weight.size(0);
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cpu_max_out", [&] {
    auto* indices_data = indices.data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.data_ptr<index_t>();

    index_t* max_indices_data = nullptr;
    int64_t max_indices_stride = 0;
    if (max_indices) {
      max_indices_data = max_indices->data_ptr<index_t>();
      max_indices_stride = max_indices->strides()[0];
    }

    auto* weight_data = weight.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    auto* bag_size_data = bag_size.data_ptr<index_t>();
    auto weight_stride0 = weight.strides()[0];
    auto weight_stride1 = weight.strides()[1];
    auto output_stride = output.strides()[0];
    int64_t numBags = bag_size.size(0);
    std::vector<bool> bag_empty(numBags, true);

    for (const auto i : c10::irange(numIndices)) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];
      TORCH_CHECK(
          word_idx >= 0 && word_idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          word_idx);
      if (word_idx != static_cast<index_t>(padding_idx)) {
        bool is_first_for_bag = bag_empty[bag];
        for (const auto dim : c10::irange(featureSize)) {
          auto& current_item = output_data[output_stride * bag + dim];
          auto weight_item =
              weight_data[weight_stride0 * word_idx + dim * weight_stride1];

          if (is_first_for_bag || (weight_item > current_item)) {
            current_item = weight_item;
            if (max_indices_data) {
              max_indices_data[max_indices_stride * bag + dim] = word_idx;
            }
          }
        }
        if (is_first_for_bag) {
          bag_empty[bag] = false;
        }
      } else {
        // Decrement bag_size to reflect that the index is padded
        bag_size_data[bag]--;
      }
    }
  });
}
```
反向
```cpp
static Tensor _embedding_bag_dense_backward_cpu_max(
    const Tensor& grad,
    const Tensor& bag_size,
    const Tensor& max_indices,
    int64_t num_weights) {
  AT_ASSERT(max_indices.defined());
  auto index_grad_weight =
      at::zeros({num_weights, grad.sizes()[1]}, grad.options());
  auto nonempty_max_indices = max_indices.index_select(0, bag_size.nonzero().view(-1));
  auto nonempty_grad = grad.index_select(0, bag_size.nonzero().view(-1));

  for (const auto dim : c10::irange(grad.sizes()[1])) {
    index_grad_weight.select(1, dim).index_add_(
      0, nonempty_max_indices.select(1, dim), nonempty_grad.select(1, dim));
  }
  return index_grad_weight;
}
```

Cuda设备
```cpp
template <typename scalar_t, typename index_t>
__global__ void EmbeddingBag_updateOutputKernel_max(
    const index_t *input, const index_t *offsets, const scalar_t *weight, scalar_t *output,
    index_t *offset2bag, int64_t numIndices, int64_t numBags,
    int64_t featureSize, int64_t weight_stride0, int64_t weight_stride1,
    index_t *bag_size, index_t *max_indices,
    index_t padding_idx, int64_t numRows) {

  // the strategy here is that each bag x feature is handled by a single thread

  int64_t chunksPerBag = ceil_div(featureSize, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < featureSize) {
      int64_t bag = chunk / chunksPerBag;
      const scalar_t *weightFeat = weight + featureDim * weight_stride1;
      int64_t begin = bag == 0 ? 0 : offsets[bag]; // forces first offset to be 0 instead of asserting on it
      int64_t end = (bag < numBags - 1) ? (offsets[bag + 1]) : numIndices;
      CUDA_KERNEL_ASSERT(end >= begin);
      scalar_t weightFeatMax = 0;
      int64_t bag_size_ = 0;
      int64_t maxWord = -1;
      for (int64_t emb = begin; emb < end; emb++) {
        bool pad = (input[emb] == padding_idx);
        CUDA_KERNEL_ASSERT(input[emb] < numRows);
        const int64_t weightRow = input[emb] * weight_stride0;
        scalar_t weightValue = weightFeat[weightRow];
        if (bag_size_ == 0 || weightValue > weightFeatMax) {
          weightFeatMax = pad ? weightFeatMax : weightValue;
          maxWord = pad ? maxWord : input[emb];
        }
        bag_size_ += pad ? 0 : 1;

        if (featureDim == 0) {
          offset2bag[emb] = bag;
        }
      }
      bag_size[bag] = bag_size_;
      max_indices[bag * featureSize + featureDim] = maxWord;
      output[bag * featureSize + featureDim] = weightFeatMax;
    }
  }
}
```
反向
```cpp
Tensor embedding_bag_backward_cuda_max(const Tensor &grad,
                                   const Tensor &max_indices,
                                   int64_t num_weights,
                                   int64_t padding_idx) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("embedding_bag_backward_cuda_max");

  auto grad_weight = at::zeros({num_weights, grad.size(1)}, grad.options());

  int64_t stride = grad_weight.stride(0);

  int64_t numBags = grad.size(0);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#if defined(USE_ROCM)
  dim3 block = dim3(64, 4);
#else
  dim3 block = dim3(32, 8);
#endif
  int grid = 1024;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "embedding_bag_backward_cuda_max", [&] {
        AT_DISPATCH_INDEX_TYPES(max_indices.scalar_type(), "embedding_bag_backward_cuda_max", [&] () {
          EmbeddingBag_accGradParametersKernel_max<
              scalar_t, index_t><<<grid, block, 0, stream>>>(
              max_indices.const_data_ptr<index_t>(), grad.const_data_ptr<scalar_t>(),
              grad_weight.mutable_data_ptr<scalar_t>(), stride, numBags,
              padding_idx, grad_weight.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  });

  return grad_weight;
}

template <typename scalar_t, typename index_t>
__global__ void EmbeddingBag_accGradParametersKernel_max(
    const index_t *max_indices, const scalar_t *gradOutput,
    scalar_t *gradWeight, int64_t stride, int64_t numBags,
    index_t padding_idx, const index_t numel) {

  using accscalar_t = acc_type<scalar_t, true>;

  int64_t chunksPerBag = ceil_div(stride, (int64_t)blockDim.x);
  int64_t numChunks = numBags * chunksPerBag;
  int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    int64_t featureDim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
    if (featureDim < stride) {
      int64_t bag = chunk / chunksPerBag;

      index_t word_idx = max_indices[bag * stride + featureDim];
      if (word_idx >= 0 && word_idx != padding_idx) {
        // If bag is empty, we have max_indices[idx] set to -1 in forward.
        fastAtomicAdd(
            gradWeight, static_cast<index_t>(word_idx * stride + featureDim),
            numel, gradOutput[bag * stride + featureDim], true);
      }
    }
  }
}
```

可以看出基本都是处理完参数后，遍历一遍取最大值，除了Max函数外还有求和、求平均等，较为繁琐

## Tensorflow_addons

API如下:
```Python
tfa.layers.EmbeddingBag(
    input_dim: int,
    output_dim: int,
    embeddings_initializer: tfa.types.Initializer = 'uniform',
    embeddings_regularizer: tfa.types.Regularizer = None,
    embeddings_constraint: tfa.types.Constraint = None,
    mask_zero: bool = False,
    combiner: str = 'sum',
    **kwargs
)
```
CPU设备
```cpp
// CPU specialization of actual computation.
template <typename T, typename Tindices>
struct EmbeddingBagFunctor<CPUDevice, T, Tindices> {
  static constexpr int64 kPacketSize = Eigen::internal::packet_traits<T>::size;
  using VectorMap = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>;
  using ConstVectorMap = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>;

  void operator()(const CPUDevice &device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::Tensor output, Combiner combiner) {
    const Eigen::Index bags = indices.dimension(0);
    const Eigen::Index sequence_length = indices.dimension(1);
    const Eigen::Index output_dim = params.dimension(1);

    const auto work = [&](Eigen::Index start, Eigen::Index end) {
      for (Eigen::Index bag = start; bag < end; ++bag) {
        VectorMap output_slice(&output(bag, 0), output_dim);
        output_slice.setZero();
        for (Eigen::Index seq = 0; seq < sequence_length; ++seq) {
          const ConstVectorMap params_slice(&params(indices(bag, seq), 0),
                                            output_dim);
          output_slice += params_slice * weights(bag, seq);
        }
        if (combiner == Combiner::kMean) {
          output_slice /= static_cast<T>(sequence_length);
        }
      }
    };

    const double bytes_loaded =
        sequence_length * (sizeof(Tindices) + sizeof(T)) +
        (sequence_length * output_dim) * sizeof(T);
    const double bytes_stored = output_dim * sizeof(T);
    const double compute_cycles =
        (sequence_length * output_dim) *
        (Eigen::TensorOpCost::AddCost<T>() + Eigen::TensorOpCost::MulCost<T>());
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, compute_cycles,
                                   /*vectorized=*/true,
                                   /*packet_size=*/kPacketSize);
    device.parallelFor(bags, cost, std::move(work));
  }
};
```
反向
```cpp
// CPU specialization of actual computation.
template <typename T, typename Tindices>
struct EmbeddingBagBackwardFunctor<CPUDevice, T, Tindices> {
  static constexpr int64 kPacketSize = Eigen::internal::packet_traits<T>::size;
  using VectorMap = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>;
  using ConstVectorMap = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>;

  void operator()(const CPUDevice &device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::ConstTensor grads,
                  typename TTypes<T, 2>::Tensor params_grads,
                  typename TTypes<T, 2>::Tensor weights_grads,
                  Combiner combiner, OpKernelContext *context) {
    const Eigen::Index sequence_length = indices.dimension(1);
    const Eigen::Index output_dim = params.dimension(1);

    std::unordered_map<Tindices, Eigen::Index> index_map;
    // The pair (x, {y_i}) in index_vec means
    // index y_i in `indices` contributes to bag `x`.
    std::vector<std::pair<Tindices, std::vector<Eigen::Index>>> index_vec;
    for (Eigen::Index i = 0; i < indices.size(); ++i) {
      Tindices index = indices.data()[i];
      if (index_map.find(index) == index_map.end()) {
        index_map[index] = index_vec.size();
        index_vec.push_back({index, {}});
      }
      index_vec[index_map[index]].second.push_back(i);
    }

    const auto compute_params_grads = [&](Eigen::Index start,
                                          Eigen::Index end) {
      for (Eigen::Index i = start; i < end; ++i) {
        VectorMap params_grads_slice(&params_grads(index_vec[i].first, 0),
                                     output_dim);
        for (Eigen::Index index : index_vec[i].second) {
          const Eigen::Index bag = index / sequence_length;
          const Eigen::Index seq = index % sequence_length;
          const ConstVectorMap grads_slice(&grads(bag, 0), output_dim);
          params_grads_slice += grads_slice * weights(bag, seq);
        }
        if (combiner == Combiner::kMean) {
          params_grads_slice /= static_cast<T>(sequence_length);
        }
      }
    };

    const Eigen::Index num_unique_params = index_vec.size();
    const double bytes_loaded = 100 * output_dim * sizeof(T);
    const double bytes_stored = output_dim * sizeof(T);
    const double compute_cycles =
        100 * output_dim *
        (Eigen::TensorOpCost::AddCost<T>() + Eigen::TensorOpCost::MulCost<T>());
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, compute_cycles,
                                   /*vectorized=*/true,
                                   /*packet_size=*/kPacketSize);
    params_grads.setZero();
    device.parallelFor(num_unique_params, cost,
                       std::move(compute_params_grads));

    const auto compute_weights_grads =
        [&](const Eigen::array<Eigen::Index, 2> &coords) -> T {
      const Eigen::Index bag = coords[0];
      const Eigen::Index seq = coords[1];
      const ConstVectorMap grads_slice(&grads(bag, 0), output_dim);
      const ConstVectorMap params_slice(&params(indices(bag, seq), 0),
                                        output_dim);
      T output = params_slice.dot(grads_slice);
      if (combiner == Combiner::kMean) {
        output /= static_cast<T>(sequence_length);
      }
      return output;
    };

    weights_grads.device(device) =
        weights_grads.generate(std::move(compute_weights_grads));
  }
};
```

Cuda设备
```cpp
// Define the GPU kernel.
template <typename T, typename Tindices, const int kThreadsPerBlock>
__global__ void EmbeddingBagGPUKernel(const Tindices *__restrict__ indices,
                                      const T *__restrict__ params,
                                      const T *__restrict__ weights,
                                      T *__restrict__ output,
                                      const Eigen::Index output_dim,
                                      const Eigen::Index sequence_length,
                                      Combiner combiner) {
  // blockIdx.x indicates which row of the output we are writing to. It also
  // indicates which `bag` we're reading from.
  // blockIdx.y indicates which chunk of that row we are writing to.
  // threadIdx.x indicates which element of that chunk we are writing to.

  // feature_idx is the position in the final dimension of the output that we
  // are writing to.
  const Eigen::Index feature_idx = blockIdx.y * kThreadsPerBlock + threadIdx.x;
  // It's necessary in case output_dim is not evenly divided by blockDim.x.
  if (feature_idx < output_dim) {
    // output_idx is the offset of the output we are writing to.
    const Eigen::Index output_idx = blockIdx.x * output_dim + feature_idx;
    // bag_offset is the offset in indices corresponding to the first
    // index of the `bag` that we will be summing over.
    const Eigen::Index bag_offset = blockIdx.x * sequence_length;
    T accum = static_cast<T>(0);
    for (Eigen::Index idx_offset = bag_offset;
         idx_offset < bag_offset + sequence_length; ++idx_offset) {
      accum += params[indices[idx_offset] * output_dim + feature_idx] *
               weights[idx_offset];
    }
    if (combiner == Combiner::kMean) {
      accum /= static_cast<T>(sequence_length);
    }
    output[output_idx] = accum;
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T, typename Tindices>
struct EmbeddingBagFunctor<GPUDevice, T, Tindices> {
  static constexpr int kThreadsPerBlock = 32;

  void operator()(const GPUDevice &device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::Tensor output, Combiner combiner) {
    const Eigen::Index bags = indices.dimension(0);
    const Eigen::Index sequence_length = indices.dimension(1);
    const Eigen::Index output_dim = params.dimension(1);

    const int blocks_per_value_vec =
        Eigen::divup(output_dim, static_cast<Eigen::Index>(kThreadsPerBlock));
    const dim3 grids = dim3(bags, blocks_per_value_vec);

    TF_CHECK_OK(GpuLaunchKernel(
        EmbeddingBagGPUKernel<T, Tindices, kThreadsPerBlock>, grids,
        kThreadsPerBlock, 0, device.stream(), indices.data(), params.data(),
        weights.data(), output.data(), output_dim, sequence_length, combiner));
  }
};
```
反向
```cpp
typedef Eigen::GpuDevice GPUDevice;

template <typename Tindices, const int kThreadsPerBlock>
__global__ void PrepTempArraysKernel(
    const Tindices *__restrict__ indices, Tindices *__restrict__ sortedIndices,
    Tindices *__restrict__ sortedIndicesCounter, const int indices_size) {
  const int arrayIdx = (blockIdx.x * kThreadsPerBlock) + threadIdx.x;
  if (arrayIdx <
      indices_size) {  // Make sure we don't run off the end of the actual array
    sortedIndices[arrayIdx] = indices[arrayIdx];
    sortedIndicesCounter[arrayIdx] = arrayIdx;
  }
}

// Define the CUDA kernel.
template <typename T, typename Tindices, const int kThreadsPerBlock>
__global__ void EmbeddingBagWeightsGradKernel(
    const int value_dim, const Tindices *__restrict__ indices,
    const T *__restrict__ values, const T *__restrict__ dloss,
    T *__restrict__ weights_grad, Combiner combiner) {
  const int sample_idx = blockIdx.x;
  const int bag_idx = blockIdx.y;
  const int bag_dim = gridDim.y;
  const int valueBaseIdx =
      indices[(sample_idx * bag_dim) + bag_idx] * value_dim;
  const int dlossBaseIdx = sample_idx * value_dim;
  // Use a full-precision accumulator even for half-precision inputs
  float partialDotProduct = 0.0f;
  for (int i = threadIdx.x; i < value_dim;
       i += blockDim.x)  // Note that some threads may stop one iteration
                         // earlier if the block straddles the end of the array
  {
    partialDotProduct +=
        static_cast<float>(values[valueBaseIdx + i] * dloss[dlossBaseIdx + i]);
  }
  unsigned activeMask = 0xffffffff;
#pragma unroll
  for (int offset = kThreadsPerBlock / 2; offset > 0; offset /= 2) {
    partialDotProduct +=
        __shfl_down_sync(activeMask, partialDotProduct, offset);
  }
  if (combiner == Combiner::kMean) {
    partialDotProduct /= static_cast<float>(bag_dim);
  }
  // Thread 0 now has the full dot product
  if (threadIdx.x == 0) {
    weights_grad[(sample_idx * bag_dim) + bag_idx] =
        static_cast<T>(partialDotProduct);
  }
}

template <typename T, typename Tindices>
__global__ void EmbeddingBagValuesGradKernel(
    const int value_dim, const int bag_dim,
    const Tindices *__restrict__ sortedIndices,
    const Tindices *__restrict__ counter, const T *__restrict__ values,
    const T *__restrict__ weights, const T *__restrict__ dloss,
    T *__restrict__ values_grad, Combiner combiner) {
  const int startIdx = blockIdx.x;
  const int chunk = blockIdx.y;
  const int kThreadsPerBlock = blockDim.x;
  const int featureIdx = threadIdx.x + (chunk * kThreadsPerBlock);
  // The core problem here is that we want to avoid parallel writes to the
  // same element of the grads. We avoid that by pre-sorting a copy of the
  // indices tensor, and also co-sorting a 'counter' array so that we still know
  // which element of the incoming gradient tensor corresponds to each. Then, we
  // take the slightly lazy approach of spinning up a warp for each element of
  // the indices array, but having each warp check the previous element before
  // it starts. If the two elements are the same, then the warp immediately
  // returns without doing anything. If not, then the warp iterates forward and
  // accumulates gradient until it hits a different index element, at which
  // point it writes the accumulated value and returns. This ensures that each
  // row of the values grad tensor is handled by one and exactly one warp.
  const int valuesIdx = ldg(sortedIndices + startIdx);
  if (startIdx > 0) {
    const int prevIdx = ldg(sortedIndices + startIdx - 1);
    if (prevIdx == valuesIdx) {
      return;  // Another block is handling this index, exit
    }
  }
  int endIdx = startIdx;
  while (endIdx < gridDim.x - 1)  // Don't run off the end of the array
  {
    int nextIdx = endIdx + 1;
    int nextValuesIdx = ldg(sortedIndices + nextIdx);
    if (nextValuesIdx == valuesIdx) {
      endIdx += 1;
    } else {
      break;
    }
  }
  if (featureIdx < value_dim)  // Don't run off the end of the row
  {
    const int outputOffset = (valuesIdx * value_dim) + featureIdx;
    float accum = 0.0f;  // Full precision even if the inputs aren't

    for (int currentIdx = startIdx; currentIdx <= endIdx; ++currentIdx) {
      int originalIdxPosition = ldg(counter + currentIdx);
      T weight = weights[originalIdxPosition];
      // The floor division on this line is correct and intentional
      T featureDloss =
          ldg(dloss + (originalIdxPosition / bag_dim) + featureIdx);
      accum += static_cast<float>(weight * featureDloss);
    }
    if (combiner == Combiner::kMean) {
      accum /= static_cast<float>(bag_dim);
    }
    values_grad[outputOffset] = static_cast<T>(accum);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T, typename Tindices>
struct EmbeddingBagBackwardFunctor<GPUDevice, T, Tindices> {
  // indices should remain unchanged, but thrust complains if it's a const
  // pointer
  void operator()(const GPUDevice &d,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::ConstTensor grads,
                  typename TTypes<T, 2>::Tensor params_grads,
                  typename TTypes<T, 2>::Tensor weights_grads,
                  Combiner combiner, OpKernelContext *context) {
    // I copy-pasted this bit from histogram_op_gpu.cu.cc and I sure hope it
    // works
    tensorflow::AllocatorAttributes gpu_allocator;
    gpu_allocator.set_on_host(false);
    gpu_allocator.set_gpu_compatible(true);

    Tensor sortedIndicesTensor;
    Tensor sortedIndicesCounterTensor;

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tindices>::value,
                                          TensorShape({indices.size()}),
                                          &sortedIndicesTensor, gpu_allocator));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<Tindices>::value,
                                TensorShape({indices.size()}),
                                &sortedIndicesCounterTensor, gpu_allocator));
    auto sortedIndices = sortedIndicesTensor.flat<Tindices>();
    auto sortedIndicesCounter = sortedIndicesCounterTensor.flat<Tindices>();
    // Note: I tried splitting the two kernels into different streams but
    // performance was barely affected.
    const Eigen::Index batch_dim = indices.dimension(0);
    const Eigen::Index bag_dim = indices.dimension(1);
    const Eigen::Index output_dim = params.dimension(1);
    const auto params_size = params.size();
    const int kThreadsPerBlock = 32;
    dim3 gridShape = dim3(batch_dim, bag_dim, 1);
    TF_CHECK_OK(GpuLaunchKernel(
        EmbeddingBagWeightsGradKernel<T, Tindices, kThreadsPerBlock>, gridShape,
        kThreadsPerBlock, 0, d.stream(), output_dim, indices.data(),
        params.data(), grads.data(), weights_grads.data(), combiner));

    const int indices_size = indices.size();
    const int values_size = params.size();
    const int total_blocks = Eigen::divup(indices_size, kThreadsPerBlock);
    gridShape = dim3(total_blocks, 1, 1);

    TF_CHECK_OK(GpuLaunchKernel(
        PrepTempArraysKernel<Tindices, kThreadsPerBlock>, gridShape,
        kThreadsPerBlock, 0, d.stream(), indices.data(), sortedIndices.data(),
        sortedIndicesCounter.data(), indices_size));

    thrust::device_ptr<Tindices> sortedIndicesCounterDevicePtr(
        sortedIndicesCounter.data());
    thrust::device_ptr<Tindices> sortedIndicesDevicePtr(sortedIndices.data());
    thrust::device_ptr<T> paramsGradDevicePtr(params_grads.data());
    thrust::fill(paramsGradDevicePtr,
                 paramsGradDevicePtr + static_cast<int>(params_size),
                 static_cast<T>(0.0f));
    thrust::sort_by_key(sortedIndicesDevicePtr,
                        sortedIndicesDevicePtr + indices_size,
                        sortedIndicesCounterDevicePtr);
    // Handle each row with as few thread blocks as possible
    int threadsPerBlock;
    int blocksPerRow;
    if (output_dim <= MAX_THREADS_PER_BLOCK) {
      blocksPerRow = 1;
      threadsPerBlock = output_dim;
    } else {
      blocksPerRow =
          Eigen::divup(static_cast<int>(output_dim), MAX_THREADS_PER_BLOCK);
      threadsPerBlock =
          Eigen::divup(static_cast<int>(output_dim), blocksPerRow);
    }
    // int blocksPerRow = 1;
    // while (threadsPerBlock > MAX_THREADS_PER_BLOCK) {
    //   threadsPerBlock = (threadsPerBlock + 1) / 2;  // Ceiling division
    //   blocksPerRow *= 2;
    // }
    gridShape = dim3(indices_size, blocksPerRow, 1);
    TF_CHECK_OK(GpuLaunchKernel(
        EmbeddingBagValuesGradKernel<T, Tindices>, gridShape, threadsPerBlock,
        0, d.stream(), output_dim, bag_dim, sortedIndices.data(),
        sortedIndicesCounter.data(), params.data(), weights.data(),
        grads.data(), params_grads.data(), combiner));
  }
};
```

后端也是类似基于C++实现


# 四、对比分析
Pytorch 在整体上功能更为丰富，如支持的 reduce 方法除了 sum、mean 之外还支持 max；max_norm 和 norm_type 可以控制输出词向量的范数；scale_grad_by_freq 可以缩放词向量权重的梯度；sparse 可以指定词向量权重梯度存储为稀疏 tensor；padding_idx 可以指定特定位置的词向量不参与梯度更新；特别地，Pytorch 的输入只能是 1D 或 2D tensor，输入是 1D 时，由 offsets 标记每个 bag 在输入中的 starting idx，输入是 2D 时，形状为 (n_bags, n_seq_len)，因此 Pytorch 支持的每个 bag 的 bag size 可以是不同的 (使用 offsets 时可以做到)。相对而言 Tensorflow_addons 只支持 sum、mean，而且 bag_size 都需要统一，同时缺少 Pytorch 的很多其他功能。因此应参照 Pytorch 的功能进行实现更为合理。


# 五、设计思路与实现方案

## 命名与参数设计
paddle.nn.functional.embedding_bag
```Python
def embedding_bag(x, weight, offsets=None, per_sample_weights=None, mode='mean', max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, padding_idx=None, include_last_offset=False, name=None):
    """
    Args:
        x(Tensor): A 1D or 2D tensor with type int32/int64, which contains the id information. If ``x`` is 1D tensor, it will be treated as the concatenation of multiple bags, and will be segmented by ``offsets`` into each bag. If ``x`` is 2D tensor, the shape should be [bag_number, sequence_length]. The value of the input id should satisfy :math: `0 <= id < params.shape[0]`.

        weight(Tensor): A tensor with shape of [num_embedding, embedding_dim] in which num_embedding indicates the size of the dictionary of embeddings and embedding_dim indicates the size of each embedding vector. Supported dtypes are int8, bfloat16, float16, float32, float64, complex64, complex128.

        offsets(Tensor, optional): Specify the starting index (in ``x``) of the sequence in each bag. Default: None.

        per_sample_weights(Tensor, optional): A tensor with the same shape of input. The variable can only be decalred when mode is set "sum". When mode is "mean", the value of weight is set to 1 by default. Default: None.

        mode(str): The string indicates calculation mode. mode can be set "sum", "mean" or "max". Default: "mean".

        max_norm(float, optional): If provided, will renormalize the embedding vectors to have a norm larger than ``max_norm`` . It will inplace update the input embedding weight in dynamic graph mode. Default: None.

        norm_type(float, optional): The p of the p-norm to compute for the max_norm option. Default: 2.0.

        scale_grad_by_freq(bool, optional): If True, the gradients of ``weight`` will be scaled by the inverse frequency of the words in the mini-batch. It is not supported when ``mode`` is "max". Default: False.

        sparse(bool, optional): If True, the gradients of ``weight`` will be a sparse tensor. It is recommended to set True because sparse update is faster. But some optimizers does not support sparse update, such as :ref:`api_paddle_optimizer_adadelta_Adadelta`, :ref:`api_paddle_optimizer_adamax_Adamax`, :ref:`api_paddle_optimizer_lamb_Lamb`. In these cases, sparse must be False. Default: False.

        padding_idx(int, optional): padding_idx needs to be in the interval [-weight.shape[0], weight.shape[0]). If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted to :math:`weight.shape[0] + padding\_idx` . It will treat the vector at ``padding_idx`` to be an all-zero padding vector. And the padding vector will not be updated while training. Default: None.

        include_last_offset(bool, optional): If True, the size of ``offsets`` will be [B+1], where B is the number of bags, and the last element will specify the ending position of the last bag. Default: False.

        name(str|None, optional): Usually name is no need to set and None by default.

    Returns:
        Tensor: The calculation of embedding params according to ``x``. The data type is the same as ``weight``.
    """
```

paddle.nn.EmbeddingBag
```Python
class EmbeddingBag(nn.Layers):
    """
    Parameters:
        num_embeddings (int): Just one element which indicate the size of the dictionary of embeddings.

        embedding_dim (int):  Just one element which indicate the size of each embedding vector respectively.

        offsets(Tensor, optional): Specify the starting index (in ``x``) of the sequence in each bag. Default: None.

        per_sample_weights(Tensor, optional): A tensor with the same shape of input. The variable can only be decalred when mode is set "sum". When mode is "mean", the value of weight is set to 1 by default. Default: None.

        mode(str): The string indicates calculation mode. mode can be set "sum", "mean" or "max". Default: "mean".

        max_norm(float, optional): If provided, will renormalize the embedding vectors to have a norm larger than ``max_norm`` . It will inplace update the input embedding weight in dynamic graph mode. Default: None.

        norm_type(float, optional): The p of the p-norm to compute for the max_norm option. Default: 2.0.

        scale_grad_by_freq(bool, optional): If True, the gradients of ``weight`` will be scaled by the inverse frequency of the words in the mini-batch. It is not supported when ``mode`` is "max". Default: False.

        sparse(bool, optional): If True, the gradients of ``weight`` will be a sparse tensor. It is recommended to set True because sparse update is faster. But some optimizers does not support sparse update, such as :ref:`api_paddle_optimizer_adadelta_Adadelta`, :ref:`api_paddle_optimizer_adamax_Adamax`, :ref:`api_paddle_optimizer_lamb_Lamb`. In these cases, sparse must be False. Default: False.

        padding_idx(int, optional): padding_idx needs to be in the interval [-weight.shape[0], weight.shape[0]). If :math:`padding\_idx < 0`, the :math:`padding\_idx` will automatically be converted to :math:`weight.shape[0] + padding\_idx` . It will treat the vector at ``padding_idx`` to be an all-zero padding vector. And the padding vector will not be updated while training. Default: None.

        include_last_offset(bool, optional): If True, the size of ``offsets`` will be [B+1], where B is the number of bags, and the last element will specify the ending position of the last bag. Default: False.

        weight_attr(ParamAttr|None, optional): To specify the weight parameter property. Default: None, which means the default weight parameter property is used. See usage for details in :ref:`api_paddle_ParamAttr` . In addition, user-defined or pre-trained word vectors can be loaded with the :attr:`param_attr` parameter. The local word vector needs to be transformed into numpy format, and the shape of local word vector should be consistent with :attr:`num_embeddings` . Then :ref:`api_paddle_nn_initializer_Assign` is used to load custom or pre-trained word vectors. See code example for details.

        name(str|None, optional): Usually name is no need to set and None by default.
    
    Attribute:
        **weight** (Parameter): the learnable weights of this layer.

    Returns:
        None
    """
    def __init__(self, num_embeddings, embedding_dim, mode='mean', max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False, padding_idx=None, include_last_offset=False, weight_attr=None, name=None):
      super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._mode = mode
        self._max_norm = max_norm
        self._norm_type = norm_type
        self._scale_grad_by_freq = scale_grad_by_freq
        self._sparse = sparse
        self._padding_idx = padding_idx
        self._include_last_offset = include_last_offset

        if self._num_embeddings <= 0:
            raise ValueError("num_embeddings must be gather than 0")

        if self._embedding_dim <= 0:
            raise ValueError("embedding_dim must be gather than 0")

        padding_idx = (
            -1
            if padding_idx is None
            else (
                padding_idx
                if padding_idx >= 0
                else (num_embeddings + padding_idx)
            )
        )

        if padding_idx >= num_embeddings or padding_idx < -num_embeddings:
            raise ValueError(
                f"padding_idx must be within [-{num_embeddings}, {num_embeddings})"
            )

        self._dtype = self._helper.get_default_dtype()
        self._size = [self._num_embeddings, self._embedding_dim]

        self._weight_attr = weight_attr
        self._name = name
        self.weight = self.create_parameter(
            attr=self._weight_attr,
            shape=self._size,
            dtype=self._dtype,
            is_bias=False,
        )

        if in_dynamic_mode() and padding_idx != -1:
            with paddle.no_grad():
                self.weight[padding_idx] = 0.0

    def forward(self, x, offsets=None, per_sample_weights=None):
        return F.embedding_bag(x, weight=self.weight, offsets=offsets, per_sample_weights=per_sample_weights, mode=self._mode, max_norm=self._max_norm, norm_type=self._norm_type, scale_grad_by_freq=self._scale_grad_by_freq, sparse=self._sparse, padding_idx=self._padding_idx, include_last_offset=self._include_last_offset, name=self._name)
```

## API 实现方案
参考[未开发完的PR](https://github.com/PaddlePaddle/Paddle/pull/49000)

1. 完善 yaml 和 infermeta 描述算子
1. 分别对 CPU 和 GPU 环境下增加 kernel 实现，需增加 mode 为 max 时的实现，以及适配更多参数支持。
1. 在 common.py 和 input.py 封装 Python API
1. 完善相关测试：动态图测试、静态图测试以及算子测试


# 六、测试和验收的考量

正确性验证：用 Embedding 以及 reduce 方法组合实现对应 EmbeddingBag 的方法，前向与反向结果对齐；

测试case：
- 测试不同输入 shape；
- 测试不同输入 dtype 类型：输入 x 可以是 'int32'，'int64'，输入 weight 可以是 'int8'， 'float16'， 'bfloat16'， 'complex64'， 'complex128'， 'float32', 'float64'；
- 测试不同设备；
- 测试动态图静态图；
- 测试不同的参数组合；

单侧文件位于：test/legacy_test/test_embeddingbag.py

# 七、可行性分析和排期规划

2024/10 完成 API 主体设计与实现；
2024/11 完成单测；

# 八、影响面

新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[Pytorch EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag)

[Tensorflow EmbeddingBag](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/EmbeddingBag)