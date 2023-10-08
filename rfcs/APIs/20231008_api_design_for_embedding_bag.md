# paddle.embedding_bag 设计文档

| API 名称     | paddle.embedding_bag                     |
| ------------ | ---------------------------------------- |
| 提交作者     | jlx                                      |
| 提交时间     | 2023-10-08                               |
| 版本号       | V1.0                                     |
| 依赖飞桨版本 | develop                                  |
| 文件名       | 20231008_api_design_for_embedding_bag.md |

# 一、概述

## 1、相关背景
`EmbeddingBag` 是 `Embedding` 的拓展，此任务的目标是在 Paddle 框架中，新增 EmbeddingBag 和 embedding_bag API，调用路径为：`paddle.nn.EmbeddingBag` 和 `paddle.nn.functional.embedding_bag`

## 2、功能目标
EmbeddingBag算子在不实例中间变量的情况下实现求和/求均值等系列运算，相比直接组合，EmbeddingBag 会有更高的计算效率和更小的内存消耗

## 3、意义
提高Embedding计算效率，使Paddle也支持EmbeddingBag


# 二、飞桨现状
目前paddle缺少相关功能实现

无类似功能API或者可组合实现方案

# 三、业内方案调研
Pytorch和Tensorflow均有embedding_bag实现，调研如下：

## PyTorch
Pytorch使用C++实现，以MAX操作为例分CPU和CUDA两份代码分别展示：

### 实现解读
CPU版本
```C++
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

Cuda版本
```C++
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
可以看出基本都是处理完参数后，遍历一遍取最大值，除了Max函数外还有求和、求平均等，较为繁琐

## Tensorflow

Tensorflow的EmbeddingBag API如下:
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
后端也是类似基于C++实现


# 四、对比分析
结论：为了性能，最终都需要Python调C++实现的CPU和CUDA版本，不能复用现有的API，需单独增加EmbeddingBag算子


# 五、设计思路与实现方案

## 命名与参数设计
仿照之前PR中的设计:

```Python
def embedding_bag(input, params, weight, mode, name=None) -> Tensor:
    """
    Args:
        input(Tensor): A tensor with type int32/int64, which contains the id information. The shape is [bag_number, sequence_length],The value of the input id should satisfy :math: `0 <= id < params.shape[0]`.

        params(Tensor): A tensor with shape of [num_embedding, embedding_dim] in which num_embedding indicates the size of the dictionary of embeddings and embedding_dim indicates the size of each embedding vector.

        weight(Tensor): A tensor with the same shape of input. The variable can only be decalred when mode is set "sum". When mode is "mean", the value of weight is set to 1 by default.

        mode(str): The string indicates calculation mode. mode can be set "sum" or "mean" for now.

        name(str|None, optional): Usually name is no need to set and None by default.
    Returns:
        Tensor: The calculation of embedding params according to 'input'. The data type is the same as 'params'.
    """
```

## API 实现方案
总体实现参考[增加C++算子教程](https://www.paddlepaddle.org.cn/documentation/docs/zh/dev_guides/api_contributing_guides/new_cpp_op_cn.html)

1. 分别对CPU和GPU环境下增加kernel实现,确保CI均编译通过
2. 在common.py和input.py封装Python API
3. 完善相关测试：动态图测试、静态图测试以及算子测试
4. 填yaml和infermeta描述算子


# 六、测试和验收的考量
1.排查出之前[PR](https://github.com/PaddlePaddle/Paddle/pull/49000)中的CI问题

2.完善其动态图测试、静态图测试以及算子测试


# 七、可行性分析和排期规划

之前已有开发者尝试开发该功能，可在其[PR](https://github.com/PaddlePaddle/Paddle/pull/49000)基础上完善，可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增 API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[Pytorch EmbeddingBag](https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#EmbeddingBag)

[Tensorflow EmbeddingBag](https://www.tensorflow.org/addons/api_docs/python/tfa/layers/EmbeddingBag)