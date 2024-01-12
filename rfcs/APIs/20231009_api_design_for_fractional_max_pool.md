# FractionalMaxPool2d / FractionalMaxPool3d API 设计文档

| API 名称 | FractionalMaxPool2d / FractionalMaxPool3d |
| - | - |
| 提交作者 | megemini(柳顺) |
| 提交时间 | 2024-01-12 |
| 版本号 | V2.1 |
| 依赖飞桨版本 | develop |
| 文件名 | 20231009_api_design_for_fractional_max_pool.md |

#### 修订记录
v2.0: 将实现方式由 python 改为 c++
v2.1: 修改接口签名

# 一、概述

## 1、相关背景

[《Fractional Max-Pooling》](https://arxiv.org/abs/1412.6071) 这篇文章介绍了一种 `fractional` 的池化方法，区别与传统的池化方法，如 `max-pooling`，`《Fractional Max-Pooling》` 的池化因子可以在 `1 < alpha < 2` 之间，也就是说，每次池化操作可以将输入缩小诸如 `sqrt(2)` 倍，而不是简单的 `2` 倍。比如，可以将输入尺寸为 `25` 缩小为输出 `18`，此时 `alpha = 25/18 = 1.39`。

文章中提到，这种池化方法可以防止传统池化方式快速缩小输入尺寸，从而影响性能的问题。可以介由网络对于更多不同尺寸输入的识别，以提升模型整体的识别能力。

飞桨目前实现了诸如 `max-pooling`、`avg-pooling` 等方法，但没有实现 `fractional max pooling`，此次实现 `fractional max pool2d / fractional max pool3d` 以提升飞桨 API 的丰富程度。

## 2、功能目标

在一个由多个通道组成的输入信号上施加分数最大池化。分数最大池化请参考论文 [《Fractional Max-Pooling》](https://arxiv.org/abs/1412.6071)
调用形式
- `paddle.nn.FractionalMaxPool2d`
- `paddle.nn.FractionalMaxPool3d`
- `paddle.nn.functional.fractional_max_pool2d`
- `paddle.nn.functional.fractional_max_pool3d`

## 3、意义

为 `Paddle` 增加 `Fractional Max-Pooling` 操作，丰富 `Paddle` 中池化操作相关的 API。

# 二、飞桨现状

飞桨目前已经提供了诸多的池化方法，如：`max_poolNd`、`avg_poolNd` 等，但尚未提供 `fractional_max_pool` 方法，底层也没有相关算子的实现。

飞桨目前将池化操作相关函数放在 `python/paddle/nn/functional/pooling.py` 文件中，另外，在 `python/paddle/nn/layer/pooling.py` 中提供了构造网络需要的模块。其中对应的 `layer` 层，均可通过调用 `functional` 相关函数实现。

由此，`paddle.nn.FractionalMaxPoolNd` 可以通过调用 `paddle.nn.functional.fractional_max_poolNd` 实现。

# 三、业内方案调研

## 算法逻辑

对比 `2*2 max pooling` (2MP) ，2MP 的采样序列为 `22222...`，如果将其中混杂 `1`，如 `1121122112...`，便可以生成 `1 < alpha = N_in/N_out < 2` 的池化结果。

因此，算法的关键是如何生成 `1121122112...` 类似的序列，以满足 `output_size` 或 `input_size * output_ratio`。

注：这里的 `1` 和 `2` 可以理解为 `kernel/pool size`，也就是每次池化的尺寸，或者是文章中的 `increments`，之所以是 `1`、`2`，前提是 `1 < alpha < 2`，也就是说，这是介于 `原尺寸` 与 `2*2 max pooling` 之间的池化操作。如果 `alpha > 2`，类似于 `3*3 max pooling`，这里的序列可以是任何大于零的整数。后续为简化谈论，假设 `1 < alpha < 2`。

文章中介绍了两种方式，`真` 随机（`random`）与 `伪` 随机（`pseudo random`）。

- `真` 随机（`random`）

    随机生成 `1` 和 `2` 的序列，只要满足：

    - 序列长度为 `output_size`
    - 序列累加和为 `input_size`

- `伪` 随机（`pseudo random`）

    这里生成的累加序列，需要满足：

    `a = ceil(alpha(i+u)), 1 < alpha = N_in/N_out < 2, 0 < u < 1, i = 0,1,2...N_out`

    长度为 `output_size + 1`，`u` 为随机数，可以利用随机种子固定住。由此生成序列：

    `diff = a[i+1] - a[i]`

生成随机序列后，便可以利用 `max` 操作，在每个池化窗口取最大值，由此产生最后的输出。

## PyTorch

`PyTorch` 底层通过 c++ 实现 `fractional_max_pool2d / fractional_max_pool3d` 函数，并通过上层的 python 对外开放相应接口。

相应的，`FractionalMaxPool2d` 通过 `fractional_max_pool2d` 实现，`FractionalMaxPool3d` 通过 `fractional_max_pool3d` 实现。

相应文档：

- [FRACTIONALMAXPOOL2D](https://pytorch.org/docs/stable/generated/torch.nn.FractionalMaxPool2d.html#fractionalmaxpool2d)
- [FRACTIONALMAXPOOL3D](https://pytorch.org/docs/stable/generated/torch.nn.FractionalMaxPool3d.html#fractionalmaxpool3d)
- [TORCH.NN.FUNCTIONAL.FRACTIONAL_MAX_POOL2D](https://pytorch.org/docs/stable/generated/torch.nn.functional.fractional_max_pool2d.html#torch.nn.functional.fractional_max_pool2d)
- [TORCH.NN.FUNCTIONAL.FRACTIONAL_MAX_POOL3D](https://pytorch.org/docs/stable/generated/torch.nn.functional.fractional_max_pool3d.html#torch.nn.functional.fractional_max_pool3d)


相应接口为：

- `torch.nn.FractionalMaxPool2d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)`

    - 文档描述
    > Applies 2D fractional max pooling over an input signal composed of several input planes.

    - 参数列表
    > kernel_size – the size of the window to take a max over.
    > output_size – the target output size
    > output_ratio – If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)
    > return_indices – if True, will return the indices along with the outputs.

    - 返回值
    > output (Tensor)

- `torch.nn.FractionalMaxPool3d(kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)`

    - 文档描述
    > Applies 3D fractional max pooling over an input signal composed of several input planes.

    - 参数列表
    > kernel_size – the size of the window to take a max over.
    > output_size – the target output size
    > output_ratio – If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)
    > return_indices – if True, will return the indices along with the outputs.

    - 返回值
    > output (Tensor)

- `torch.nn.functional.fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)`

    - 文档描述
    > Applies 2D fractional max pooling over an input signal composed of several input planes.

    - 参数列表
    > kernel_size – the size of the window to take a max over.
    > output_size – the target output size
    > output_ratio – If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)
    > return_indices – if True, will return the indices along with the outputs.

    - 返回值
    > output (Tensor)

- `torch.nn.functional.fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)`

    - 文档描述
    > Applies 3D fractional max pooling over an input signal composed of several input planes.

    - 参数列表
    > kernel_size – the size of the window to take a max over.
    > output_size – the target output size
    > output_ratio – If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)
    > return_indices – if True, will return the indices along with the outputs.

    - 返回值
    > output (Tensor)


实现逻辑：

由于 `fractional_max_pool2d` 与 `fractional_max_pool3d` 最大的区别是维度，其他逻辑基本相同，所以，后续以 `fractional_max_pool2d` 为主要分析对象。

相关源代码涉及文件：

- `torch/nn/functional.py` *
- `torch/csrc/api/include/torch/nn/options/pooling.h`
- `torch/csrc/api/include/torch/nn/functional/pooling.h` *
- `torch/csrc/api/include/torch/nn/modules/pooling.h`
- `torch/csrc/api/src/nn/modules/pooling.cpp`
- `aten/src/ATen/native/FractionalMaxPooling.h` *
- `aten/src/ATen/native/FractionalMaxPool2d.cpp` *

这里只分析上述带有 `*` 的主要源文件。

- `torch/nn/functional.py`

    这里对 `fractional_max_pool2d` 开放 API：

    ``` python
    def fractional_max_pool2d_with_indices(
        input: Tensor, kernel_size: BroadcastingList2[int],
        output_size: Optional[BroadcastingList2[int]] = None,
        output_ratio: Optional[BroadcastingList2[float]] = None,
        return_indices: bool = False,
        _random_samples: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        r"""
        fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None)

        ...
        """
        if has_torch_function_variadic(input, _random_samples):
            return handle_torch_function(
                fractional_max_pool2d_with_indices,
                (input, _random_samples),
                input,
                kernel_size,
                output_size=output_size,
                output_ratio=output_ratio,
                return_indices=return_indices,
                _random_samples=_random_samples,
            )
        if output_size is None and output_ratio is None:
            raise ValueError("fractional_max_pool2d requires specifying either " "an output_size or an output_ratio")
        if output_size is None:
            assert output_ratio is not None
            if len(output_ratio) > 2:
                raise ValueError("fractional_max_pool2d requires output_ratio to either be a single Int or tuple of Ints.")
            _output_ratio = _pair(output_ratio)
            output_size = [int(input.size(-2) * _output_ratio[0]), int(input.size(-1) * _output_ratio[1])]

        if _random_samples is None:
            n_batch = 1 if input.dim() == 3 else input.size(0)
            _random_samples = torch.rand(n_batch, input.size(-3), 2, dtype=input.dtype, device=input.device)
        return torch._C._nn.fractional_max_pool2d(input, kernel_size, output_size, _random_samples)


    def _fractional_max_pool2d(
        input: Tensor, kernel_size: BroadcastingList2[int],
        output_size: Optional[BroadcastingList2[int]] = None,
        output_ratio: Optional[BroadcastingList2[float]] = None,
        return_indices: bool = False,
        _random_samples: Optional[Tensor] = None
    ) -> Tensor:
        if has_torch_function_variadic(input, _random_samples):
            return handle_torch_function(
                fractional_max_pool2d,
                (input, _random_samples),
                input,
                kernel_size,
                output_size=output_size,
                output_ratio=output_ratio,
                return_indices=return_indices,
                _random_samples=_random_samples,
            )
        return fractional_max_pool2d_with_indices(
            input, kernel_size, output_size, output_ratio, return_indices, _random_samples
        )[0]


    fractional_max_pool2d = boolean_dispatch(
        arg_name="return_indices",
        arg_index=4,
        default=False,
        if_true=fractional_max_pool2d_with_indices,
        if_false=_fractional_max_pool2d,
        module_name=__name__,
        func_name="fractional_max_pool2d",
    )
    ```

    这里根据是否需要 `indices` 对接口进行分发，最终都是调用 `fractional_max_pool2d_with_indices`。


- `torch/csrc/api/include/torch/nn/functional/pooling.h`

    上面的接口会调用这里对应的 c++ 实现：

    ``` cpp
    namespace detail {
    inline std::tuple<Tensor, Tensor> fractional_max_pool2d_with_indices(
        const Tensor& input,
        const ExpandingArray<2>& kernel_size,
        const c10::optional<ExpandingArray<2>>& output_size,
        const c10::optional<ExpandingArray<2, double>>& output_ratio,
        const Tensor& _random_samples) {
    if (output_size == c10::nullopt && output_ratio == c10::nullopt) {
        TORCH_CHECK(
            false,
            "fractional_max_pool2d requires specifying either ",
            "an output_size or an output_ratio");
    }
    c10::optional<ExpandingArray<2>> output_size_ = output_size;
    if (output_size_ == c10::nullopt) {
        TORCH_INTERNAL_ASSERT(output_ratio != c10::nullopt);
        output_size_ = {
            (int64_t)(static_cast<double>(input.size(-2)) * (*output_ratio.value())[0]),
            (int64_t)(static_cast<double>(input.size(-1)) * (*output_ratio.value())[1])};
    }

    Tensor _random_samples_ = _random_samples;
    if (!_random_samples_.defined()) {
        auto n_batch = input.dim() == 3 ? 1 : input.size(0);
        _random_samples_ = torch::rand(
            {n_batch, input.size(-3), 2},
            torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    }
    return torch::fractional_max_pool2d(
        input, kernel_size, *output_size_, _random_samples_);
    }
    } // namespace detail
    ```

    这里是 `fractional_max_pool2d` 主要入口，主要做了以下几处处理：

    - 如果没有 `output_size`，根据 `output_ratio` 生成 `output_size`
    - 如果没有 `_random_samples`，根据输入的维度生成随机序列
    - 调用主要方法 `torch::fractional_max_pool2d(input, kernel_size, *output_size_, _random_samples_);}`


- `aten/src/ATen/native/FractionalMaxPool2d.cpp`

    这里实现了具体的逻辑：

    ``` cpp
    template <typename scalar_t>
    static void fractional_max_pool2d_out_single_batch_frame(
    scalar_t* input,
    scalar_t* output,
    int64_t* indices,
    scalar_t* randomSamples,
    int numPlanes,
    int inputW, int inputH,
    int outputW, int outputH,
    int poolSizeW, int poolSizeH) {
    at::parallel_for(0, numPlanes, 0, [&](int64_t start, int64_t end) {
        for (const auto plane : c10::irange(start, end)) {
        /* each plane contains 2 random samples, one for W and one for H */
        scalar_t* randomSamplesForPlane = randomSamples + plane * 2;

        /* Generate interval sequence */
        auto sequenceW = generate_intervals<scalar_t>(
            randomSamplesForPlane[0], inputW, outputW, poolSizeW);
        auto sequenceH = generate_intervals<scalar_t>(
            randomSamplesForPlane[1], inputH, outputH, poolSizeH);

        /* loop over output */
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        int h, w;

        scalar_t* inputForPlane = input + plane * inputW * inputH;
        scalar_t* outputForPlane = output + plane * outputW * outputH;
        int64_t* indicesForPlane = indices + plane * outputW * outputH;

        for (h = 0; h < outputH; ++h) {
            int inputHStart = sequenceH[h];

            for (w = 0; w < outputW; ++w) {
            int inputWStart = sequenceW[w];

            int h2 = inputHStart, w2 = inputWStart;
            scalar_t maxVal = -std::numeric_limits<scalar_t>::infinity();
            int64_t maxIndex = h2 * inputW + w2;

            for (h2 = inputHStart; h2 < inputHStart + poolSizeH; ++h2) {
                for (w2 = inputWStart; w2 < inputWStart + poolSizeW; ++w2) {
                AT_ASSERT(h2 >= 0 && h2 < inputH);
                AT_ASSERT(w2 >= 0 && w2 < inputW);

                int planeIndex = h2 * inputW + w2;
                scalar_t val = inputForPlane[planeIndex];
                if (val > maxVal || std::isnan(val)) {
                    maxVal = val;
                    maxIndex = planeIndex;
                }
                }
            }

            outputForPlane[h * outputW + w] = maxVal;
            indicesForPlane[h * outputW + w] = maxIndex;
            }
        }
        }
    });
    }    
    ```

    此文件实现了 `fractional_max_pool2d` 的主要逻辑，上面只摘抄了最关键的代码。
    
    主要逻辑为：

    - 生成采样的序列
    - 获取序列中的每个 pool 中的最大值

    其中，生成采样序列的逻辑在 `aten/src/ATen/native/FractionalMaxPooling.h`：

    ``` cpp
    template<typename scalar_t>
    static inline std::vector<int> generate_intervals(
        scalar_t sample,
        int64_t inputSize,
        int64_t outputSize,
        int64_t poolSize) {
        std::vector<int> sequence(outputSize);
        if (outputSize > 1) {
            scalar_t alpha = static_cast<scalar_t>(inputSize - poolSize) /
            static_cast<scalar_t>(outputSize - 1);

            for (const auto i : c10::irange(outputSize - 1)) {
                sequence[i] =
                    static_cast<int>((i + sample) * alpha) - static_cast<int>(sample * alpha);
            }
        }
        if (outputSize > 0) {
            sequence[outputSize - 1] = inputSize - poolSize;
        }
        return sequence;
    }
    ```

从上面的源代码分析可以看到，`PyTorch` 对于 `fractional_max_pool` 只实现了 `pseudo random` 的方式，而没有 `random` 的方式。


## TensorFlow

`TensorFlow` 实现了 `tf.nn.fractional_max_pool` 函数，对应 `PyTorch` 的函数为 `fractional_max_pool2d`。

相应的，实现了 `tf.raw_ops.FractionalMaxPool` ，对应 `PyTorch` 的 `FractionalMaxPool2d`。

`TensorFlow` 并没有 `3D` 相关的实现。

`3D` 相对 `2D` ，多了一个 `depth` 或者 `time` 等类似的维度。

相应文档：

- [tf.raw_ops.FractionalMaxPool](https://tensorflow.google.cn/api_docs/python/tf/raw_ops/FractionalMaxPool?hl=en)
- [tf.nn.fractional_max_pool](https://tensorflow.google.cn/api_docs/python/tf/nn/fractional_max_pool?hl=en)

相应接口为：

- `tf.raw_ops.FractionalMaxPool`

    - 文档描述
    > Performs fractional max pooling on the input.

    - 参数列表
    > value – A Tensor. 4-D with shape [batch, height, width, channels].
    > pooling_ratio – An int or list of ints that has length 1, 2 or 4. 
    > pseudo_random – An optional bool. Defaults to False. When set to True, generates the pooling sequence in a pseudorandom fashion, otherwise, in a random fashion.
    > overlapping – An optional bool. Defaults to False. When set to True, it means when pooling, the values at the boundary of adjacent pooling cells are used by both cells.
    > deterministic – 	An optional bool. Defaults to False. When set to True, a fixed pooling region will be used when iterating over a FractionalMaxPool node in the computation graph.
    > seed – An optional int. Defaults to 0. If set to be non-zero, the random number generator is seeded by the given seed. Otherwise it is seeded by a random seed.
    > seed2 – An optional int. Defaults to 0. An second seed to avoid seed collision.
    > name – A name for the operation (optional).

    - 返回值
    > output (A tuple of Tensor objects)

- `tf.nn.fractional_max_pool`

    - 文档描述
    > Performs fractional max pooling on the input.

    - 参数列表
    > value – A Tensor. 4-D with shape [batch, height, width, channels].
    > pooling_ratio – An int or list of ints that has length 1, 2 or 4. 
    > pseudo_random – An optional bool. Defaults to False. When set to True, generates the pooling sequence in a pseudorandom fashion, otherwise, in a random fashion.
    > overlapping – An optional bool. Defaults to False. When set to True, it means when pooling, the values at the boundary of adjacent pooling cells are used by both cells.
    > seed – An optional int. Defaults to 0. If set to be non-zero, the random number generator is seeded by the given seed. Otherwise it is seeded by a random seed.
    > name – A name for the operation (optional).

    - 返回值
    > output (A tuple of Tensor objects)

实现逻辑：

相关源代码涉及文件：

- `tensorflow/python/ops/nn_ops.py` *
- `tensorflow/core/kernels/fractional_pool_common.h`
- `tensorflow/core/kernels/fractional_pool_common.cc` *
- `tensorflow/core/kernels/fractional_max_pool_op.cc` *

这里只分析上述带有 `*` 的主要源文件。

- `tensorflow/python/ops/nn_ops.py`

    这里注册 python 接口：

    ``` python
    @tf_export("nn.fractional_max_pool", v1=[])
    @dispatch.add_dispatch_support
    def fractional_max_pool_v2(value,
                            pooling_ratio,
                            pseudo_random=False,
                            overlapping=False,
                            seed=0,
                            name=None):  # pylint: disable=redefined-builtin
    if (isinstance(pooling_ratio, (list, tuple))):
        if (pooling_ratio[0] != 1.0 or pooling_ratio[-1] != 1.0):
        raise ValueError(
            "`pooling_ratio` should have first and last elements with value 1.0. "
            f"Received: pooling_ratio={pooling_ratio}")
        for element in pooling_ratio:
        if element < 1.0:
            raise ValueError(
                f"`pooling_ratio` elements should be >= 1.0. "
                f"Received: pooling_ratio={pooling_ratio}")
    elif (isinstance(pooling_ratio, (int, float))):
        if pooling_ratio < 1.0:
        raise ValueError(
            "`pooling_ratio` should be >= 1.0. "
            f"Received: pooling_ratio={pooling_ratio}")
    else:
        raise ValueError(
            "`pooling_ratio` should be an int or a list of ints. "
            f"Received: pooling_ratio={pooling_ratio}")

    pooling_ratio = _get_sequence(pooling_ratio, 2, 3, "pooling_ratio")

    if seed == 0:
        if config.is_op_determinism_enabled():
        raise ValueError(
            f"tf.nn.fractional_max_pool requires a non-zero seed to be passed in "
            f"when determinism is enabled, but got seed={seed}. Please pass in a "
            f'non-zero seed, e.g. by passing "seed=1".')
        return gen_nn_ops.fractional_max_pool(value, pooling_ratio, pseudo_random,
                                            overlapping, deterministic=False,
                                            seed=0, seed2=0, name=name)
    else:
        seed1, seed2 = random_seed.get_seed(seed)
        return gen_nn_ops.fractional_max_pool(value, pooling_ratio, pseudo_random,
                                            overlapping, deterministic=True,
                                            seed=seed1, seed2=seed2, name=name)
    
    ```

    可以看到，与 `PyTorch` 不同的是，`TensorFlow` 多了几个参数：
    
    - `overlapping` 控制 pool 边界是否计算在内
    - `pseudo_random` 是否是伪随机
    - `seed` 随机种子

- `tensorflow/core/kernels/fractional_max_pool_op.cc`

    这里实现了主要逻辑：

    ``` cpp
    template <typename T>
    class FractionalMaxPoolOp : public OpKernel {
    public:
    explicit FractionalMaxPoolOp(OpKernelConstruction* context)
        : OpKernel(context) {
        
        ...

        if (deterministic_) {
        // If both seeds are not set when deterministic_ is true, force set seeds.
        if ((seed_ == 0) && (seed2_ == 0)) {
            seed_ = random::New64();
            seed2_ = random::New64();
        }
        } else {
        OP_REQUIRES(
            context, (seed_ == 0) && (seed2_ == 0),
            errors::InvalidArgument(
                "Both seed and seed2 should be 0 if deterministic is false."));
        }
    }

    void Compute(OpKernelContext* context) override {
        typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
            ConstEigenMatrixMap;
        typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
            EigenMatrixMap;

        constexpr int tensor_in_and_out_dims = 4;

        const Tensor& tensor_in = context->input(0);

        std::vector<int> input_size(tensor_in_and_out_dims);
        std::vector<int> output_size(tensor_in_and_out_dims);
        for (int i = 0; i < tensor_in_and_out_dims; ++i) {
            input_size[i] = tensor_in.dim_size(i);
        }
        // Output size.
        for (int i = 0; i < tensor_in_and_out_dims; ++i) {
            // This must match the same logic in the shape function in
            // core/ops/nn_ops.cc.
            output_size[i] =
                static_cast<int>(std::floor(input_size[i] / pooling_ratio_[i]));
            DCHECK_GT(output_size[i], 0);
        }

        // Generate pooling sequence.
        std::vector<int64_t> height_cum_seq;
        std::vector<int64_t> width_cum_seq;
        GuardedPhiloxRandom generator;
        generator.Init(seed_, seed2_);
        height_cum_seq = GeneratePoolingSequence(input_size[1], output_size[1],
                                                &generator, pseudo_random_);
        width_cum_seq = GeneratePoolingSequence(input_size[2], output_size[2],
                                                &generator, pseudo_random_);

        // Prepare output.
        Tensor* output_tensor = nullptr;
        Tensor* output_height_seq_tensor = nullptr;
        Tensor* output_width_seq_tensor = nullptr;

        ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), input_size[3],
                                input_size[2] * input_size[1] * input_size[0]);

        EigenMatrixMap out_mat(output_tensor->flat<T>().data(), output_size[3],
                            output_size[2] * output_size[1] * output_size[0]);

        // Initializes the output tensor with MIN<T>.
        output_tensor->flat<T>().setConstant(Eigen::NumTraits<T>::lowest());

        auto output_height_seq_flat = output_height_seq_tensor->flat<int64_t>();
        auto output_width_seq_flat = output_width_seq_tensor->flat<int64_t>();

        // Set output tensors.
        for (int i = 0; i < height_cum_seq.size(); ++i) {
            output_height_seq_flat(i) = height_cum_seq[i];
        }

        for (int i = 0; i < width_cum_seq.size(); ++i) {
            output_width_seq_flat(i) = width_cum_seq[i];
        }

        // For both input and output,
        // 0: batch
        // 1: height / row
        // 2: width / col
        // 3: depth / channel
        const int64_t height_max = input_size[1] - 1;
        const int64_t width_max = input_size[2] - 1;
        for (int64_t b = 0; b < input_size[0]; ++b) {
            // height sequence.
            for (int64_t hs = 0; hs < height_cum_seq.size() - 1; ++hs) {
                // height start and end.
                const int64_t height_start = height_cum_seq[hs];
                int64_t height_end =
                    overlapping_ ? height_cum_seq[hs + 1] : height_cum_seq[hs + 1] - 1;
                height_end = std::min(height_end, height_max);

                // width sequence.
                for (int64_t ws = 0; ws < width_cum_seq.size() - 1; ++ws) {
                    const int64_t out_offset =
                        (b * output_size[1] + hs) * output_size[2] + ws;
                    // width start and end.
                    const int64_t width_start = width_cum_seq[ws];
                    int64_t width_end =
                        overlapping_ ? width_cum_seq[ws + 1] : width_cum_seq[ws + 1] - 1;
                    width_end = std::min(width_end, width_max);
                    for (int64_t h = height_start; h <= height_end; ++h) {
                        for (int64_t w = width_start; w <= width_end; ++w) {
                        const int64_t in_offset =
                            (b * input_size[1] + h) * input_size[2] + w;
                        out_mat.col(out_offset) =
                            out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
                        }
                    }
                }
            }
        }
    }

    };    
    ```

    其中序列生成的函数在： `tensorflow/core/kernels/fractional_pool_common.cc`

    ``` cpp
    static std::vector<int64_t> GeneratePoolingSequencePseudoRandom(
        int input_length, int output_length, GuardedPhiloxRandom* generator) {
        std::vector<int64_t> cum_seq(output_length + 1, 0);
        std::vector<int64_t> diff(output_length, 0);

        double alpha = static_cast<double>(input_length) / output_length;
        int k = input_length / output_length;

        double u_max1 = (k + 2) / alpha - 1;
        double u_max2 = (input_length + 1 - k) / alpha - (output_length - 1);
        double max_u = std::min(u_max1, u_max2);

        // Generate random number in parallel.
        auto local_gen = generator->ReserveSamples32(2);
        random::SimplePhilox random(&local_gen);
        const double u = random.RandDouble() * max_u;

        cum_seq[0] = 1;
        cum_seq[output_length] = input_length + 1;
        for (int i = 1; i < output_length; ++i) {
            cum_seq[i] = static_cast<int>(ceil(alpha * (i + u)));
        }

        for (int i = 0; i < output_length; ++i) {
            diff[i] = cum_seq[i + 1] - cum_seq[i];
        }

        return diff;
    }

    static std::vector<int64_t> GeneratePoolingSequenceRandom(
        int input_length, int output_length, GuardedPhiloxRandom* generator) {
        int k = input_length / output_length;
        int num_random_spot = input_length % output_length;
        std::vector<int64_t> diff(output_length, k);

        for (int i = 0; i < num_random_spot; ++i) {
            diff[i] += 1;
        }

        // Randomly shuffle this vector.
        auto local_gen = generator->ReserveSamples32(diff.size());
        random::SingleSampleAdapter<random::PhiloxRandom> single(&local_gen);
        const auto uniform = [&single](uint32 n) { return single() % n; };
        RandomShuffle(diff.begin(), diff.end(), uniform);

        return diff;
    }

    std::vector<int64_t> GeneratePoolingSequence(int input_length,
                                                int output_length,
                                                GuardedPhiloxRandom* generator,
                                                bool pseudo_random) {
        std::vector<int64_t> diff;
        // This is a case that regular pooling can handle, just return diff with
        // each element input_length/output_length.
        if (input_length % output_length == 0) {
            diff = std::vector<int64_t>(output_length, input_length / output_length);
        }

        if (pseudo_random) {
            diff = GeneratePoolingSequencePseudoRandom(input_length, output_length,
                                                    generator);
        } else {
            diff =
                GeneratePoolingSequenceRandom(input_length, output_length, generator);
        }

        // Sanity check.
        int k = input_length / output_length;
        for (int i = 0; i < output_length; ++i) {
            // k<= diff[i] <= k+1.
            DCHECK_GE(diff[i], k);
            DCHECK_LE(diff[i], k + 1);
        }

        // Return cumulative sequence.
        std::vector<int64_t> cum_seq(output_length + 1, 0);
        for (int i = 1; i < cum_seq.size(); ++i) {
            cum_seq[i] = cum_seq[i - 1] + diff[i - 1];
        }
        return cum_seq;
    }
    ```

    这里根据 `pseudo_random` 标记为生成 `伪` 随机序列，或者 `真` 随机序列。


# 四、对比分析

抛开 `PyTorch` 与 `TensorFlow` 对于 API 的组织方式不同来说，两者：

相同：

- `PyTorch` 与 `TensorFlow` 都实现了 `fractional_max_pool` 函数。
- `PyTorch` 与 `TensorFlow` 都是通过底层 c++ 实现具体逻辑，并通过 python 公开 API。

不同：

- `PyTorch` 实现了 `2D` 与 `3D` 两种维度的函数，`TensorFlow` 只有 `2D` 这种维度（`channel` 不算在内）。
- `TensorFlow` 有 `真` 随机与 `伪` 随机两种序列生成方式，`PyTorch` 只有 `伪` 随机一种。
- `TensorFlow` 的实现更接近文章中的描述

    这是 `PyTorch` 与 `TensorFlow` 最大的不同点。文章中的 `fractional` 根据 `N_in/N_out` 得出，也就是说，只需要这两个参数即可。
    `PyTorch` 提供了 `kernel_size`、`output_size`、`output_ratio` 这三个参数，这三个参数都可以影响 `N_in/N_out`，这更像是传统池化的方法。
    `TensorFlow` 只提供了 `pooling_ratio`，利用这个参数即可得到 `N_out`，而且提供了 `overlapping` 参数，利用这个参数可以影响 `kernel_size`。而且，由此可以看出，`TensorFlow` 实现的 `fractional max pooling` 更具有一般性，而 `adaptive max pooling` 则可以看作 `fractional max pooling` 的一种特例。
    `PyTorch` 只利用随机序列作为 stride，而不是同时将其作为 kernel 进行池化，`TensorFlow` 将随机序列既作为 stride 同时也作为 kernel 进行池化，更符合论文中的描述方式，所以，这里以 `TensorFlow` 的方式进行实现。

    - `fractional max pooling` : `a = ceiling(alpha(i+u)), 1 < alpha = N_in/N_out < 2, 0 < u < 1`
    - `adaptive max pooling` : `a = ceiling(alpha(i+1)), 1 < alpha = N_in/N_out < 2`

另外，两者都有反向梯度的计算（由于不影响主要逻辑分析，且代码较多，上述代码分析没有具体列出）。

由于飞桨已经实现了 `AdaptiveMaxPool1D / AdaptiveMaxPool2D / AdaptiveMaxPool3D`，其签名为：

- `paddle.nn.AdaptiveMaxPool1D(output_size, return_mask=False, name=None)`

为了保持一致性，这里也只使用 `output_size` 一个必要参数，实现方法更接近文章以及 `TensorFlow`。


# 五、设计思路与实现方案

本方案共涉及三部分：

- 命名与参数设计 (python API) : `paddle.nn.functional.fractional_max_pool2d`, `paddle.nn.functional.fractional_max_pool3d`
- 底层 OP 设计
- python layer 实现 : `paddle.nn.FractionalMaxPool2d`, `paddle.nn.FractionalMaxPool3d`

由于 `fractional max pooling` 与 `adaptive max pooling` 接口特性较为相似，后续设计方案以 `共用 adaptive max pooling 底层算子` 为主要设计思路。

## 命名与参数设计 (python API)

涉及文件：`python/paddle/nn/functional/pooling.py`

添加 python 上层接口:

- `paddle.nn.functional.fractional_max_pool2d`
- `paddle.nn.FractionalMaxPool2d`

    ``` python
    paddle.nn.functional.fractional_max_pool2d(
        x:Tensor,
        output_size:Union[int, list, tuple], 
        kernel_size:Optional[Union[int, list, tuple]]=None,
        random_u:Optional[float]=None,
        return_mask:bool=False,
        name:str=None)
    ```

    - 参数列表
    > x (Tensor) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > output_size (int|list|tuple) – 输出的尺寸。
    > kernel_size (int|list|tuple, optional) – 核大小。
    > random_u (float, optional) – 随机序列所需随机数。
    > return_mask (bool, optional) – 是否返回最大值的索引。
    > name (str, optional) – 操作名称。

    - 返回值
    > Tensor, return_mask=False
    > Tensor and mask, return_mask=True

- `paddle.nn.functional.fractional_max_pool3d`
- `paddle.nn.FractionalMaxPool3d`
    
    ``` python
    paddle.nn.functional.fractional_max_pool3d(
        x:Tensor,
        output_size:Union[int, list, tuple], 
        kernel_size:Optional[Union[int, list, tuple]]=None,
        random_u:Optional[float]=None,
        return_mask:bool=False,
        name:str=None)
    ```

    - 参数列表
    > x (Tensor) – 输入的一个 Tensor。数据类型支持：float32、float64、int32、int64。
    > output_size (int|list|tuple) – 输出的尺寸。
    > kernel_size (int|list|tuple, optional) – 核大小。
    > random_u (float, optional) – 随机序列所需随机数。
    > return_mask (bool, optional) – 是否返回最大值的索引。
    > name (str, optional) – 操作名称。

    - 返回值
    > Tensor, return_mask=False
    > Tensor and mask, return_mask=True

这里重点分析 `paddle.nn.functional.fractional_max_poolNd` 接口的命名与参数设计，`paddle.nn.FractionalMaxPoolNd` 与之类似。

*注意* ： 相较 v1.0 版本的设计文档，这里简化了较多的参数，特说明如下：

- 不使用 `data_format`

    分析目前 pooling 接口主要源文件 `python/paddle/nn/functional/pooling.py`，以 `max_pool2d` 为例：

    - 主要涉及两个底层算子： `max_pool2d_with_index` 和 `pool2d`
    - 其中 `max_pool2d_with_index` 可以返回 `mask`，`pool2d` 不可以返回 `mask`
    - 其中 `max_pool2d_with_index` 不支持 `data_format`，`pool2d` 支持 `data_format`

    因此，当使用 `return_mask` 返回 `mask` 时，`data_format must be set to NCHW`。
    没有一个算子能够完整支持这两个参数，这是目前 pooling 底层算子较大的矛盾。

    由于设计方案以 `共用 adaptive max pooling 底层算子` 为主要设计思路，所以，这里参考 `adaptive max pooling` 的接口：

    `adaptive_max_pool2d(x, output_size, return_mask=False, name=None)`

    不使用 `data_format` 参数。

- 移除 `pseudo_random`, `overlapping`, `seed` 

    参考 `PyTorch` 的设计方案，这里将只使用 `伪` 随机的方式生成池化序列，并在 c++ 算子内部实现。

*注意* ： 相较 v2.0 版本的设计文档，这里增加多个参数，特说明如下：

- `kernel_size`

    此参数默认为 `None`，表示使用 `disjoint（non-overlapping）` 模式。
    当此参数不为 `None` 时，使用 `overlapping` 模式，与 PyTorch 的实现保持一致。此处参考 Fractional Max-Pool 作者 Benjamin Graham 的解释：

    > Hello. My original implementation (for sparse ConvNets) generated regions using this code:https://github.com/btgraham/SparseConvNet-archived/blob/bdde325c28f64b895cebfdbe301a2ddca7870174/SparseConvNet/Regions.cu#L31

    并与作者提供的代码保持一致。

- `random_u`

    增加随机序列所需的随机数参数，以方便进行复现。

## 底层 OP 设计

> *注意* 以下具体实现以实际代码为准。

涉及文件：

- `paddle/phi/api/yaml/ops.yaml` 算子描述及定义

    ``` yaml
    - op : max_pool2d_with_index
    args : (Tensor x, int[] kernel_size, int[] strides= {1, 1}, int[] paddings = {0, 0}, bool global_pooling = false, bool adaptive = false, bool fractional = false)
    output : Tensor(out), Tensor(mask)
    infer_meta :
        func : MaxPoolWithIndexInferMeta
    kernel :
        func : max_pool2d_with_index
    backward : max_pool2d_with_index_grad

    - op : max_pool3d_with_index
    args : (Tensor x, int[] kernel_size, int[] strides = {1, 1, 1}, int[] paddings = {0, 0, 0}, bool global_pooling = false, bool adaptive = false, bool fractional = false)
    output : Tensor(out), Tensor(mask)
    infer_meta :
        func : MaxPoolWithIndexInferMeta
    kernel :
        func : max_pool3d_with_index
    backward : max_pool3d_with_index_grad    
    ```
    
    增加 `bool` 类型 `fractional` 参数，默认为 `false`

- `paddle/phi/api/yaml/backward.yaml` 算子描述及定义

    ``` yaml
    - backward_op : max_pool2d_with_index_grad
    forward : max_pool2d_with_index(Tensor x, int[] kernel_size, int[] strides = {1, 1}, int[] paddings = {0, 0}, bool global_pooling = false, bool adaptive = false, bool fractional = false) -> Tensor(out), Tensor(mask)
    args : (Tensor x, Tensor mask, Tensor out_grad, int[] kernel_size, int[] strides, int[] paddings, bool global_pooling, bool adaptive, bool fractional)
    output : Tensor(x_grad)
    infer_meta :
        func : MaxPoolWithIndexGradInferMeta
    kernel :
        func : max_pool2d_with_index_grad

    - backward_op : max_pool3d_with_index_grad
    forward : max_pool3d_with_index(Tensor x, int[] kernel_size, int[] strides = {1, 1, 1}, int[] paddings = {0, 0, 0}, bool global_pooling = false, bool adaptive = false, bool fractional = false) -> Tensor(out), Tensor(mask)
    args : (Tensor x, Tensor mask, Tensor out_grad, int[] kernel_size, int[] strides, int[] paddings, bool global_pooling, bool adaptive, bool fractional)
    output : Tensor(x_grad)
    infer_meta :
        func : MaxPoolWithIndexGradInferMeta
    kernel :
        func : max_pool3d_with_index_grad    
    ```

    增加 `bool` 类型 `fractional` 参数，默认为 `false`

- `paddle/phi/infermeta/unary.h` 算子 InferMeta

    ``` cpp
    void MaxPoolWithIndexInferMeta(const MetaTensor& x,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                bool global_pooling,
                                bool adaptive,
                                bool fractional,
                                MetaTensor* out,
                                MetaTensor* mask,
                                MetaConfig config = MetaConfig());
    ```
    增加 `fractional` 参数

- `paddle/phi/infermeta/unary.cc`

    ``` cpp
    void MaxPoolWithIndexInferMeta(const MetaTensor& x,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                bool global_pooling,
                                bool adaptive,
                                bool fractional,
                                MetaTensor* out,
                                MetaTensor* mask,
                                MetaConfig config) {
    ...
    if (adaptive || fractional) {
        output_shape.insert(
            output_shape.end(), kernel_size_.begin(), kernel_size_.end());
    } else {
        ...
    }
    ...
    }
    ```

    增加 `fractional` 参数，并且，与 `adaptive` 一样，共用 `kernel_size_` 参数，此参数在此实际为 `output_size`。

- `paddle/phi/infermeta/backward.h`

    ``` cpp
    void MaxPoolWithIndexGradInferMeta(const MetaTensor& x,
                                   const MetaTensor& mask,
                                   const MetaTensor& dout,
                                   const std::vector<int>& kernel_size,
                                   const std::vector<int>& strides,
                                   const std::vector<int>& paddings,
                                   bool global_pooling,
                                   bool adaptive,
                                   bool fractional,
                                   MetaTensor* dx);
    ```

    增加 `fractional` 参数。

- `paddle/phi/infermeta/backward.cc`

    ``` cpp
    void MaxPoolWithIndexGradInferMeta(const MetaTensor& x,
                                    const MetaTensor& mask,
                                    const MetaTensor& dout,
                                    const std::vector<int>& kernel_size,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    bool global_pooling,
                                    bool adaptive,
                                    bool fractional,
                                    MetaTensor* dx) {
    dx->share_meta(x);
    }
    ```

    增加 `fractional` 参数。

- `paddle/phi/kernels/pool_kernel.h` 算子 Kernel

    ``` cpp
    template <typename T, typename Context>
    void MaxPool2dWithIndexKernel(const Context& ctx,
                                const DenseTensor& x,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                bool global_pooling,
                                bool adaptive,
                                bool fractional,
                                DenseTensor* out,
                                DenseTensor* mask);

    template <typename T, typename Context>
    void MaxPool3dWithIndexKernel(const Context& ctx,
                                const DenseTensor& x,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                bool global_pooling,
                                bool adaptive,
                                bool fractional,
                                DenseTensor* out,
                                DenseTensor* mask);
    ```

    增加 `fractional` 参数。


- `paddle/phi/kernels/funcs/pooling.h`

    ``` cpp
    template <typename Context, typename T1, typename T2>
    class MaxPool2dWithIndexFunctor {
    public:
    void operator()(const Context& context,
                    const DenseTensor& input,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* output,
                    DenseTensor* mask);
    };

    template <typename Context, typename T1, typename T2>
    class MaxPool2dWithIndexGradFunctor {
    public:
    void operator()(const Context& context,
                    const DenseTensor& output_grad,
                    const DenseTensor& mask,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* input_grad);
    };

    template <typename Context, typename T1, typename T2>
    class MaxPool3dWithIndexFunctor {
    public:
    void operator()(const Context& context,
                    const DenseTensor& input,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* output,
                    DenseTensor* mask);
    };

    template <typename Context, typename T1, typename T2>
    class MaxPool3dWithIndexGradFunctor {
    public:
    void operator()(const Context& context,
                    const DenseTensor& output_grad,
                    const DenseTensor& mask,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* input_grad);
    };
    ```

    增加 `fractional` 参数。

    ``` cpp
    HOSTDEVICE inline int FractionalStartIndex()
    HOSTDEVICE inline int FractionalEndIndex()
    ```

    生成池化序列的方法。

- `paddle/phi/kernels/impl/pool_kernel_impl.h`

    ``` cpp
    template <typename Context, typename T1, typename T2 = int>
    void MaxPoolWithIndexRawKernel(const Context& ctx,
                                const DenseTensor& x,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                bool global_pooling,
                                bool adaptive,
                                bool fractional,
                                DenseTensor* out,
                                DenseTensor* mask) {
    std::vector<int> paddings_ = paddings;
    std::vector<int> kernel_size_ = kernel_size;

    if (global_pooling) {
        for (size_t i = 0; i < kernel_size_.size(); ++i) {
        paddings_[i] = 0;
        kernel_size_[i] = static_cast<int>(x.dims()[i + 2]);
        }
    }

    switch (kernel_size_.size()) {
        case 2: {
        funcs::MaxPool2dWithIndexFunctor<Context, T1, T2> pool2d_forward;
        pool2d_forward(ctx,
                        x,
                        kernel_size_,
                        strides,
                        paddings_,
                        adaptive,
                        fractional,
                        out,
                        mask);
        } break;
        case 3: {
        funcs::MaxPool3dWithIndexFunctor<Context, T1, T2> pool3d_forward;
        pool3d_forward(ctx,
                        x,
                        kernel_size_,
                        strides,
                        paddings_,
                        adaptive,
                        fractional,
                        out,
                        mask);
        } break;
        default: {
        PADDLE_THROW(
            errors::InvalidArgument("Pool op only supports 2D and 3D input."));
        }
    }
    }

    template <typename T, typename Context>
    void MaxPool2dWithIndexKernel(const Context& ctx,
                                const DenseTensor& x,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                bool global_pooling,
                                bool adaptive,
                                bool fractional,
                                DenseTensor* out,
                                DenseTensor* mask) {
    MaxPoolWithIndexRawKernel<Context, T>(ctx,
                                            x,
                                            kernel_size,
                                            strides,
                                            paddings,
                                            global_pooling,
                                            adaptive,
                                            fractional,
                                            out,
                                            mask);
    }

    template <typename T, typename Context>
    void MaxPool3dWithIndexKernel(const Context& ctx,
                                const DenseTensor& x,
                                const std::vector<int>& kernel_size,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                bool global_pooling,
                                bool adaptive,
                                bool fractional,
                                DenseTensor* out,
                                DenseTensor* mask) {
    MaxPoolWithIndexRawKernel<Context, T>(ctx,
                                            x,
                                            kernel_size,
                                            strides,
                                            paddings,
                                            global_pooling,
                                            adaptive,
                                            fractional,
                                            out,
                                            mask);
    }

    ```

    增加 `fractional` 参数，分发方法时带上 `fracional`。

- `paddle/phi/kernels/pool_grad_kernel.h` 反向算子

    ``` cpp
    template <typename T, typename Context>
    void MaxPool2dWithIndexGradKernel(const Context& ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& mask,
                                    const DenseTensor& dout,
                                    const std::vector<int>& kernel_size,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    bool global_pooling,
                                    bool adaptive,
                                    bool fracional,
                                    DenseTensor* dx);

    template <typename T, typename Context>
    void MaxPool3dWithIndexGradKernel(const Context& ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& mask,
                                    const DenseTensor& dout,
                                    const std::vector<int>& kernel_size,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    bool global_pooling,
                                    bool adaptive,
                                    bool fracional,
                                    DenseTensor* dx);

    ```

    增加 `fractional` 参数。

- `paddle/phi/kernels/impl/pool_grad_kernel_impl.h`

    ``` cpp
    template <typename Context, typename T1, typename T2 = int>
    void MaxPoolWithIndexGradRawKernel(const Context& ctx,
                                    const DenseTensor& x UNUSED,
                                    const DenseTensor& mask,
                                    const DenseTensor& dout,
                                    const std::vector<int>& kernel_size,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    bool global_pooling,
                                    bool adaptive,
                                    bool fractional,
                                    DenseTensor* dx) {
    std::vector<int> paddings_ = paddings;
    std::vector<int> kernel_size_ = kernel_size;

    if (global_pooling) {
        for (size_t i = 0; i < kernel_size_.size(); ++i) {
        paddings_[i] = 0;
        kernel_size_[i] = static_cast<int>(dx->dims()[i + 2]);
        }
    }

    if (dx) {
        ctx.template Alloc<T1>(dx);
        funcs::set_constant(ctx, dx, 0);

        switch (kernel_size_.size()) {
        case 2: {
            funcs::MaxPool2dWithIndexGradFunctor<Context, T1, T2> pool2d_backward;
            pool2d_backward(ctx,
                            dout,
                            mask,
                            kernel_size_,
                            strides,
                            paddings_,
                            adaptive,
                            fractional,
                            dx);
        } break;
        case 3: {
            funcs::MaxPool3dWithIndexGradFunctor<Context, T1, T2> pool3d_backward;
            pool3d_backward(ctx,
                            dout,
                            mask,
                            kernel_size_,
                            strides,
                            paddings_,
                            adaptive,
                            fractional,
                            dx);
        } break;
        default: {
            PADDLE_THROW(
                errors::InvalidArgument("Pool op only supports 2D and 3D input."));
        }
        }
    }
    }

    template <typename T, typename Context>
    void MaxPool2dWithIndexGradKernel(const Context& ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& mask,
                                    const DenseTensor& dout,
                                    const std::vector<int>& kernel_size,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    bool global_pooling,
                                    bool adaptive,
                                    bool fractional,
                                    DenseTensor* dx) {
    MaxPoolWithIndexGradRawKernel<Context, T>(ctx,
                                                x,
                                                mask,
                                                dout,
                                                kernel_size,
                                                strides,
                                                paddings,
                                                global_pooling,
                                                adaptive,
                                                fractional,
                                                dx);
    }

    template <typename T, typename Context>
    void MaxPool3dWithIndexGradKernel(const Context& ctx,
                                    const DenseTensor& x,
                                    const DenseTensor& mask,
                                    const DenseTensor& dout,
                                    const std::vector<int>& kernel_size,
                                    const std::vector<int>& strides,
                                    const std::vector<int>& paddings,
                                    bool global_pooling,
                                    bool adaptive,
                                    bool fractional,
                                    DenseTensor* dx) {
    MaxPoolWithIndexGradRawKernel<Context, T>(ctx,
                                                x,
                                                mask,
                                                dout,
                                                kernel_size,
                                                strides,
                                                paddings,
                                                global_pooling,
                                                adaptive,
                                                fractional,
                                                dx);
    }
    
    ```

    增加 `fractional` 参数，分发方法时带上 `fracional`。

- `paddle/phi/kernels/funcs/pooling.cc` 算子 CPU 实现

    ``` cpp
    template <typename T1, typename T2>
    class MaxPool2dWithIndexFunctor<CPUContext, T1, T2> {
    public:
    void operator()(const CPUContext& context,
                    const DenseTensor& input,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* output,
                    DenseTensor* mask) {
    ...

        int hstart = 0, hend = 0;
        int wstart = 0, wend = 0;
        for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
            for (int ph = 0; ph < output_height; ++ph) {
            if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
            } else if (fractional) {
                // TODO(megemini)
            } else {
                hstart = ph * stride_height - padding_height;
                hend = std::min(hstart + ksize_height, input_height);
                hstart = std::max(hstart, 0);
            }
            for (int pw = 0; pw < output_width; ++pw) {
                if (adaptive) {
                wstart = AdaptStartIndex(pw, input_width, output_width);
                wend = AdaptEndIndex(pw, input_width, output_width);
                } else if (fractional) {
                // TODO(megemini)
                } else {
                wstart = pw * stride_width - padding_width;
                wend = std::min(wstart + ksize_width, input_width);
                wstart = std::max(wstart, 0);
                }

        ...
            }
            }
        ...
        }
        }
    }
    };

    /*
    * All tensors are in NCHW format.
    * Ksize, strides, paddings are two elements. These two elements represent
    * height and width, respectively.
    */
    template <typename T1, typename T2>
    class MaxPool2dWithIndexGradFunctor<CPUContext, T1, T2> {
    public:
    void operator()(const CPUContext& context,
                    const DenseTensor& output_grad,
                    const DenseTensor& mask,
                    const std::vector<int>& ksize UNUSED,
                    const std::vector<int>& strides UNUSED,
                    const std::vector<int>& paddings UNUSED,
                    bool adaptive UNUSED,
                    bool fractional UNUSED,
                    DenseTensor* input_grad) {
    };}

    /*
    * All tensors are in NCDHW format.
    * Ksize, strides, paddings are three elements. These three elements represent
    * depth, height and width, respectively.
    */
    template <typename T1, typename T2>
    class MaxPool3dWithIndexFunctor<CPUContext, T1, T2> {
    public:
    void operator()(const CPUContext& context,
                    const DenseTensor& input,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* output,
                    DenseTensor* mask) {
        ...

        int dstart = 0, dend = 0;
        int hstart = 0, hend = 0;
        int wstart = 0, wend = 0;
        for (int i = 0; i < batch_size; i++) {
        for (int c = 0; c < output_channels; ++c) {
            for (int pd = 0; pd < output_depth; ++pd) {
            if (adaptive) {
                dstart = AdaptStartIndex(pd, input_depth, output_depth);
                dend = AdaptEndIndex(pd, input_depth, output_depth);
            } else if (fractional) {
                /* TODO(megemini) */
            } else {
                dstart = pd * stride_depth - padding_depth;
                dend = std::min(dstart + ksize_depth, input_depth);
                dstart = std::max(dstart, 0);
            }
            for (int ph = 0; ph < output_height; ++ph) {
                if (adaptive) {
                hstart = AdaptStartIndex(ph, input_height, output_height);
                hend = AdaptEndIndex(ph, input_height, output_height);
                } else if (fractional) {
                /* TODO(megemini) */
                } else {
                hstart = ph * stride_height - padding_height;
                hend = std::min(hstart + ksize_height, input_height);
                hstart = std::max(hstart, 0);
                }
                for (int pw = 0; pw < output_width; ++pw) {
                if (adaptive) {
                    wstart = AdaptStartIndex(pw, input_width, output_width);
                    wend = AdaptEndIndex(pw, input_width, output_width);
                } else if (fractional) {
                    // TODO(megemini)
                } else {
                    wstart = pw * stride_width - padding_width;
                    wend = std::min(wstart + ksize_width, input_width);
                    wstart = std::max(wstart, 0);
                }

        ...
            }
            }
        ...
        }
        }
    }
    };}

    /*
    * All tensors are in NCDHW format.
    * Ksize, strides, paddings are three elements. These three elements represent
    * depth, height and width, respectively.
    */
    template <typename T1, typename T2>
    class MaxPool3dWithIndexGradFunctor<CPUContext, T1, T2> {
    public:
    void operator()(const CPUContext& context,
                    const DenseTensor& output_grad,
                    const DenseTensor& mask,
                    const std::vector<int>& ksize UNUSED,
                    const std::vector<int>& strides UNUSED,
                    const std::vector<int>& paddings UNUSED,
                    bool adaptive UNUSED,
                    bool fractional UNUSED,
                    DenseTensor* input_grad) {
    };}

    ```

    这里实现主要的 cpu 算子的逻辑（正向与反向），通过 `fractional` 参数生成池化序列，主要逻辑与 `adaptive` 相似。
    
    这里没有 `data_format` 参数的设计，建议后续能够统一 `poolNd` 与 `max_poolNd` 的算子实现。

    另外，这里需要再增加一个 `0 < random < 1` 的随机数，以生成 `伪` 随机池化序列，这个随机数需要可以通过 `paddle.seed` 固定住。

- `paddle/phi/kernels/funcs/pooling.cu` 算子 GPU 实现

    ``` cpp
    template <typename T1, typename T2>
    __global__ void KernelMaxPool2dWithIdx(const int nthreads,
                                        const T1* input_data,
                                        const int channels,
                                        const int input_height,
                                        const int input_width,
                                        const int output_height,
                                        const int output_width,
                                        const int ksize_height,
                                        const int ksize_width,
                                        const int stride_height,
                                        const int stride_width,
                                        const int padding_height,
                                        const int padding_width,
                                        bool adaptive,
                                        bool fractional,
                                        T1* output_data,
                                        T2* mask_data,
                                        FastDivModForPooling divmods) 

    template <typename T1, typename T2>
    __global__ void KernelMaxPool2DWithIdxGrad(const int nthreads,
                                            const T1* output_grad,
                                            const T2* mask_data,
                                            const int channels,
                                            const int input_height,
                                            const int input_width,
                                            const int output_height,
                                            const int output_width,
                                            const int ksize_height,
                                            const int ksize_width,
                                            const int stride_height,
                                            const int stride_width,
                                            const int padding_height,
                                            const int padding_width,
                                            bool adaptive,
                                            bool fractional,
                                            T1* input_grad,
                                            FastDivModForPooling divmods) 
    /*
    * All tensors are in NCHW format.
    * Ksize, strides, paddings are two elements. These two elements represent
    * height and width, respectively.
    */
    template <typename T1, typename T2>
    class MaxPool2dWithIndexFunctor<phi::GPUContext, T1, T2> {
    public:
    void operator()(const phi::GPUContext& context,
                    const DenseTensor& input,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* output,
                    DenseTensor* mask)
    };

    /*
    * All tensors are in NCHW format.
    * Ksize, strides, paddings are two elements. These two elements represent
    * height and width, respectively.
    */
    template <typename T1, typename T2>
    class MaxPool2dWithIndexGradFunctor<phi::GPUContext, T1, T2> {
    public:
    void operator()(const phi::GPUContext& context,
                    const DenseTensor& output_grad,
                    const DenseTensor& mask,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* input_grad) 
    };


    template <typename T1, typename T2>
    __global__ void KernelMaxPool3DWithIdx(const int ncd,
                                        const T1* input_data,
                                        const int channels,
                                        const int input_depth,
                                        const int input_height,
                                        const int input_width,
                                        const int output_depth,
                                        const int output_height,
                                        const int output_width,
                                        const int ksize_depth,
                                        const int ksize_height,
                                        const int ksize_width,
                                        const int stride_depth,
                                        const int stride_height,
                                        const int stride_width,
                                        const int padding_depth,
                                        const int padding_height,
                                        const int padding_width,
                                        bool adaptive,
                                        bool fractional,
                                        T1* output_data,
                                        T2* mask_data,
                                        FastDivModForPooling3D divmods_output) 

    template <typename T1, typename T2>
    __global__ void KernelMaxPool3DWithIdxGrad(
                        const int ncd,
                        const T1* output_grad,
                        const T2* mask,
                        const int channels,
                        const int input_depth,
                        const int input_height,
                        const int input_width,
                        const int output_depth,
                        const int output_height,
                        const int output_width,
                        const int ksize_depth,
                        const int ksize_height,
                        const int ksize_width,
                        const int stride_depth,
                        const int stride_height,
                        const int stride_width,
                        const int padding_depth,
                        const int padding_height,
                        const int padding_width,
                        bool adaptive,
                        bool fractional,
                        T1* input_grad,
                        FastDivModForPooling3D divmods_output) 
        
    /*
    * All tensors are in NCDHW format.
    * Ksize, strides, paddings are three elements. These three elements represent
    * depth, height and width, respectively.
    */
    template <typename T1, typename T2>
    class MaxPool3dWithIndexFunctor<phi::GPUContext, T1, T2> {
    public:
    void operator()(const phi::GPUContext& context,
                    const DenseTensor& input,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* output,
                    DenseTensor* mask)
    };

    /*
    * All tensors are in NCDHW format.
    * Ksize, strides, paddings are three elements. These three elements represent
    * depth, height and width, respectively.
    */
    template <typename T1, typename T2>
    class MaxPool3dWithIndexGradFunctor<phi::GPUContext, T1, T2> {
    public:
    void operator()(const phi::GPUContext& context,
                    const DenseTensor& output_grad,
                    const DenseTensor& mask,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& paddings,
                    bool adaptive,
                    bool fractional,
                    DenseTensor* input_grad)
    };

    ```

    主要逻辑与 CPU 算子类似，这里不再赘述，有一个需要单独指出的是，PR：https://github.com/PaddlePaddle/Paddle/pull/45959 中，单独针对 `AdaptiveKernelMaxPool2dWithIdx` 做了优化，本次设计方案暂不进行优化方面的设计。

### 池化序列的生成方法

这里编写了一个简化的程序，以演示如何生成 fractional 的池化序列：

``` cpp
#include <iostream>
#include <math.h>

inline int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<float>(ph * input_size) / output_size));
}

inline int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<float>((ph + 1) * input_size) / output_size));
}

inline int FractionalStartIndex(int ph, double alpha, double u) {
  return static_cast<int>(
      // subtract `1` for index from `0`
      ceil(alpha * (ph + u) - 1));
}

inline int FractionalEndIndex(int ph, double alpha, double u) {
  return static_cast<int>(
      // subtract `1` for index from `0`
      ceil(alpha * (ph + 1 + u) - 1)) ;
}


int main()
{
    int input_height = 32;
    int output_height = 25;
    for (int ph = 0; ph < output_height; ++ph) {

        int hstart = AdaptStartIndex(ph, input_height, output_height);
        int hend = AdaptEndIndex(ph, input_height, output_height);

        std::cout << "------------" << std::endl;
        std::cout << "ph " << ph << std::endl;
        std::cout << "hstart " << hstart << " hend " << hend << " diff " << hend - hstart << std::endl;
    }

    std::cout << "====================" << std::endl;

    double alpha = static_cast<double>(input_height) / output_height;
    int base = input_height / output_height;

    double u_max1 = (base + 2) / alpha - 1;
    double u_max2 = (input_height + 1 - base) / alpha - (output_height - 1);
    double max_u = std::min(u_max1, u_max2);

    double u = 0.8 * max_u;

    for (int ph = 0; ph < output_height; ++ph) {

        int hstart = FractionalStartIndex(ph, alpha, u);
        int hend = FractionalEndIndex(ph, alpha, u);
        hend = std::min(hend, input_height);

        std::cout << "------------" << std::endl;
        std::cout << "ph " << ph << std::endl;
        std::cout << "hstart " << hstart << " hend " << hend << " diff " << hend - hstart << std::endl;
    }

    std::cout << "alpha is " << alpha << " u is " << u << " max u is " << max_u << std::endl;
}

```

运行后得到结果：

``` shell
$> g++ n38_index.cc -Wall && ./a.out
------------
ph 0
hstart 0 hend 2 diff 2
------------
ph 1
hstart 1 hend 3 diff 2
------------
ph 2
hstart 2 hend 4 diff 2
------------
ph 3
hstart 3 hend 6 diff 3
------------
ph 4
hstart 5 hend 7 diff 2
------------
ph 5
hstart 6 hend 8 diff 2
------------
ph 6
hstart 7 hend 9 diff 2
------------
ph 7
hstart 8 hend 11 diff 3
------------
ph 8
hstart 10 hend 12 diff 2
------------
ph 9
hstart 11 hend 13 diff 2
------------
ph 10
hstart 12 hend 15 diff 3
------------
ph 11
hstart 14 hend 16 diff 2
------------
ph 12
hstart 15 hend 17 diff 2
------------
ph 13
hstart 16 hend 18 diff 2
------------
ph 14
hstart 17 hend 20 diff 3
------------
ph 15
hstart 19 hend 21 diff 2
------------
ph 16
hstart 20 hend 22 diff 2
------------
ph 17
hstart 21 hend 24 diff 3
------------
ph 18
hstart 23 hend 25 diff 2
------------
ph 19
hstart 24 hend 26 diff 2
------------
ph 20
hstart 25 hend 27 diff 2
------------
ph 21
hstart 26 hend 29 diff 3
------------
ph 22
hstart 28 hend 30 diff 2
------------
ph 23
hstart 29 hend 31 diff 2
------------
ph 24
hstart 30 hend 32 diff 2
====================
------------
ph 0
hstart 1 hend 2 diff 1
------------
ph 1
hstart 2 hend 3 diff 1
------------
ph 2
hstart 3 hend 4 diff 1
------------
ph 3
hstart 4 hend 6 diff 2
------------
ph 4
hstart 6 hend 7 diff 1
------------
ph 5
hstart 7 hend 8 diff 1
------------
ph 6
hstart 8 hend 9 diff 1
------------
ph 7
hstart 9 hend 11 diff 2
------------
ph 8
hstart 11 hend 12 diff 1
------------
ph 9
hstart 12 hend 13 diff 1
------------
ph 10
hstart 13 hend 15 diff 2
------------
ph 11
hstart 15 hend 16 diff 1
------------
ph 12
hstart 16 hend 17 diff 1
------------
ph 13
hstart 17 hend 18 diff 1
------------
ph 14
hstart 18 hend 20 diff 2
------------
ph 15
hstart 20 hend 21 diff 1
------------
ph 16
hstart 21 hend 22 diff 1
------------
ph 17
hstart 22 hend 24 diff 2
------------
ph 18
hstart 24 hend 25 diff 1
------------
ph 19
hstart 25 hend 26 diff 1
------------
ph 20
hstart 26 hend 27 diff 1
------------
ph 21
hstart 27 hend 29 diff 2
------------
ph 22
hstart 29 hend 30 diff 1
------------
ph 23
hstart 30 hend 31 diff 1
------------
ph 24
hstart 31 hend 32 diff 1
alpha is 1.28 u is 0.8 max u is 1

```

可以看到
- adaptive 的池化序列为 `2...3...` 的样式，fractional 的池化序列为 `1...2...` 的样式。
- adaptive 的池化序列存在 index 交叉，而 fractional 不存在交叉。

另外:
- `FractionalStrartIndex` 和 `FractionalEndIndex` 需要减去 `1`，因为根据论文中的算法要求，使用 `ceil`，将使 index 从 `1` 开始，所以这里需要减去 `1`。
- `hend = std::min(hend, input_height);` 这里需要与 input 比对取小值，同样是由于 `ceil` 导致。

## python layer 实现

涉及文件：

- `python/paddle/nn/layer/pooling.py`

    ``` python
    class FractionalMaxPool2D(Layer):
    """
    TODO(megemini)
    """

        def __init__(self, output_size, kernel_size=None, random_u=None, return_mask=False, name=None):
            super().__init__()
            ...

    class FractionalMaxPool3D(Layer):
    """
    TODO(megemini)
    """

        def __init__(self, output_size, kernel_size=None, random_u=None, return_mask=False, name=None):
            super().__init__()
            ...
    ```

    主要通过调用相应的方法实现。

# 六、测试和验收的考量

测试考虑的case如下：

- **编程范式场景**
  - 常规覆盖动态图和静态图的测试场景
  - 需要测试 C++ 算子
  - 需要测试 python 接口

- **硬件场景**
  常规需覆盖 CPU、GPU 两种测试场景

- **参数组合场景**
  - 需要测试 2D / 3D 两类接口
  - 需要测试 1 < N_in/N_out < 2, N_in/N_out > 2 的情况
  - 需要测试 output_size 为 int/list/tuple 的情况
  - 需要测试 return_mask
  - 需要测试 不同数据类型的场景
  - 需要异常测试，如 N_in/N_out < 1

- **计算精度**
  需要保证 `前向/后向` 计算的精度正确性，通过 numpy 实现的函数的对比结果

- **维度测试**
  - 需要测试 2D / 3D 两类接口

# 七、可行性分析及规划排期

- 每个接口开发约 7 个工作日
- 每个接口测试约 3 个工作日

计划 3～4 周的工作量可以完成接口的开发预测是。

# 八、影响面

无其他影响。

# 名词解释

无

# 附件及参考资料

- [《Fractional Max-Pooling》](https://arxiv.org/abs/1412.6071)
- [FRACTIONALMAXPOOL2D](https://pytorch.org/docs/stable/generated/torch.nn.FractionalMaxPool2d.html#fractionalmaxpool2d)
- [FRACTIONALMAXPOOL3D](https://pytorch.org/docs/stable/generated/torch.nn.FractionalMaxPool3d.html#fractionalmaxpool3d)
- [TORCH.NN.FUNCTIONAL.FRACTIONAL_MAX_POOL2D](https://pytorch.org/docs/stable/generated/torch.nn.functional.fractional_max_pool2d.html#torch.nn.functional.fractional_max_pool2d)
- [TORCH.NN.FUNCTIONAL.FRACTIONAL_MAX_POOL3D](https://pytorch.org/docs/stable/generated/torch.nn.functional.fractional_max_pool3d.html#torch.nn.functional.fractional_max_pool3d)
- [tf.raw_ops.FractionalMaxPool](https://tensorflow.google.cn/api_docs/python/tf/raw_ops/FractionalMaxPool?hl=en)
- [tf.nn.fractional_max_pool](https://tensorflow.google.cn/api_docs/python/tf/nn/fractional_max_pool?hl=en)
