# paddle.histogramdd设计文档

| API 名称     | paddle.histogramdd                  |
| ------------ | ----------------------------------- |
| 提交作者     | coco                                |
| 提交时间     | 2023-10-01                          |
| 版本号       | V1.0                                |
| 依赖飞桨版本 | develop                             |
| 文件名       | 20231001_api_defign_for_histogramdd |

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，Paddle需要扩充API，调用路径为：

- paddle.histogramdd

## 2、功能目标

实现多维的histogram直方图计算

## 3、意义

飞桨支持为多维tensor进行直方图计算

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## PyTorch

PyTorch中有API `torch.histogramdd(input, bins, *, range=None, weight=None, density=False, out=None) -> (Tensor, Tensor[])` 

## 实现方法

从实现方法上，PyTorch是通过c++实现的，[代码位置](https://github.com/pytorch/pytorch/blob/e4414716d55c14d86298d6434e23d47605532cca/aten/src/ATen/native/cpu/HistogramKernel.cpp#L207-L255)

```cpp
/* Some pre- and post- processing steps for the main algorithm.
 * Initializes hist to 0, calls into the main algorithm, and normalizes output if necessary.
 */
template<BIN_SELECTION_ALGORITHM bin_algorithm>
void histogramdd_out_cpu_template(const Tensor& self, const c10::optional<Tensor>& weight, bool density,
        Tensor& hist, const TensorList& bin_edges) {
    hist.fill_(0);

    const int64_t N = self.size(-1);
    const int64_t M = std::accumulate(self.sizes().begin(), self.sizes().end() - 1,
            (int64_t)1, std::multiplies<int64_t>());

    const Tensor reshaped_input = self.reshape({M, N});

    const auto reshaped_weight = weight.has_value()
            ? c10::optional<Tensor>(weight.value().reshape({M}))
            : c10::optional<Tensor>();

    std::vector<Tensor> bin_edges_contig(bin_edges.size());
    for (const auto dim : c10::irange(bin_edges_contig.size())) {
        bin_edges_contig[dim] = bin_edges[dim].contiguous();
    }

    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, self.scalar_type(), "histogram_cpu", [&]() {
        histogramdd_cpu_contiguous<scalar_t, bin_algorithm>(
                hist, bin_edges_contig, reshaped_input, reshaped_weight);
    });

    /* Divides each bin's value by the total count/weight in all bins,
     * and by the bin's volume.
     */
    if (density) {
        const auto hist_sum = hist.sum().item();
        hist.div_(hist_sum);

         /* For each dimension, divides each bin's value
          * by the bin's length in that dimension.
          */
        for (const auto dim : c10::irange(N)) {
            const auto bin_lengths = bin_edges[dim].diff();

            // Used to reshape bin_lengths to align with the corresponding dimension of hist.
            std::vector<int64_t> shape(N, 1);
            shape[dim] = bin_lengths.numel();

            hist.div_(bin_lengths.reshape(shape));
        }
    }
}
```

具体在`histogramdd_cpu_contiguous`中的实现，[代码位置](https://github.com/pytorch/pytorch/blob/4c3d3b717619d9d09e95878c1f01bfb5d4fee0d0/aten/src/ATen/native/cpu/HistogramKernel.cpp#L79C3-L205)

```cpp
template<typename input_t, BIN_SELECTION_ALGORITHM algorithm>
void histogramdd_cpu_contiguous(Tensor& hist, const TensorList& bin_edges,
        const Tensor& input, const c10::optional<Tensor>& weight) {
    TORCH_INTERNAL_ASSERT(input.dim() == 2);

    const int64_t N = input.size(0);
    if (weight.has_value()) {
        TORCH_INTERNAL_ASSERT(weight.value().dim() == 1 && weight.value().numel() == N);
    }

    const int64_t D = input.size(1);
    TORCH_INTERNAL_ASSERT(int64_t(bin_edges.size()) == D);
    for (const auto dim : c10::irange(D)) {
        TORCH_INTERNAL_ASSERT(bin_edges[dim].is_contiguous());
        TORCH_INTERNAL_ASSERT(hist.size(dim) + 1 == bin_edges[dim].numel());
    }

    if (D == 0) {
        // hist is an empty tensor in this case; nothing to do here
        return;
    }

    TensorAccessor<input_t, 2> accessor_in = input.accessor<input_t, 2>();

    /* Constructs a c10::optional<TensorAccessor> containing an accessor iff
     * the optional weight tensor has a value.
     */
    const auto accessor_wt = weight.has_value()
            ? c10::optional<TensorAccessor<input_t, 1>>(weight.value().accessor<input_t, 1>())
            : c10::optional<TensorAccessor<input_t, 1>>();

    std::vector<input_t*> bin_seq(D);
    std::vector<int64_t> num_bin_edges(D);
    std::vector<input_t> leftmost_edge(D), rightmost_edge(D);

    for (const auto dim : c10::irange(D)) {
        bin_seq[dim] = bin_edges[dim].data_ptr<input_t>();
        num_bin_edges[dim] = bin_edges[dim].numel();
        leftmost_edge[dim] = bin_seq[dim][0];
        rightmost_edge[dim] = bin_seq[dim][num_bin_edges[dim] - 1];
    }

    int64_t GRAIN_SIZE = std::max(int64_t(1), HISTOGRAM_GRAIN_SIZE / D);

    /* Parallelizes processing of input using at::parallel_for.
     * Each thread accumulates a local result into their own slice of
     * thread_histograms which get summed together at the end.
     */
    const auto num_threads = at::get_num_threads();
    const auto hist_sizes = hist.sizes();
    DimVector thread_hist_sizes(hist_sizes.size() + 1);
    thread_hist_sizes[0] = num_threads;
    std::copy(hist_sizes.begin(), hist_sizes.end(),
              thread_hist_sizes.begin() + 1);
    Tensor thread_histograms = at::zeros(thread_hist_sizes, hist.dtype());
    TORCH_INTERNAL_ASSERT(thread_histograms.is_contiguous());

    at::parallel_for(0, N, GRAIN_SIZE, [&](int64_t start, int64_t end) {
        const auto tid = at::get_thread_num();
        auto hist_strides = thread_histograms.strides();
        input_t *hist_local_data = thread_histograms.data_ptr<input_t>();

        // View only this thread's local results
        hist_local_data += hist_strides[0] * tid;
        hist_strides = hist_strides.slice(1);

        for (const auto i : c10::irange(start, end)) {
            bool skip_elt = false;
            int64_t hist_index = 0;

            for (const auto dim : c10::irange(D)) {
                const input_t elt = accessor_in[i][dim];

                // Skips elements which fall outside the specified bins and NaN elements
                if (!(elt >= leftmost_edge[dim] && elt <= rightmost_edge[dim])) {
                    skip_elt = true;
                    break;
                }

                int64_t pos = -1;

                if (algorithm == BINARY_SEARCH) {
                    // Handles the general case via binary search on the bin edges.
                    pos = std::upper_bound(bin_seq[dim], bin_seq[dim] + num_bin_edges[dim], elt)
                            - bin_seq[dim] - 1;
                } else if (algorithm == LINEAR_INTERPOLATION
                        || algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
                    /* When bin_edges is known to be a linear progression, maps elt to
                     * the appropriate bin via simple division.
                     */
                    pos = static_cast<int64_t>((elt - leftmost_edge[dim])
                            * (num_bin_edges[dim] - 1)
                            / (rightmost_edge[dim] - leftmost_edge[dim]));

                    /* Ensures consistency with bin_edges by checking the bins to the left and right
                     * of the selected position. Necessary for cases in which an element very close
                     * to a bin edge may be misclassified by simple division.
                     */
                    if (algorithm == LINEAR_INTERPOLATION_WITH_LOCAL_SEARCH) {
                        int64_t pos_min = std::max(static_cast<int64_t>(0), pos - 1);
                        int64_t pos_max = std::min(pos + 2, num_bin_edges[dim]);
                        pos = std::upper_bound(bin_seq[dim] + pos_min, bin_seq[dim] + pos_max, elt)
                                - bin_seq[dim] - 1;
                    }
                } else {
                    TORCH_INTERNAL_ASSERT(false);
                }

                // Unlike other bins, the rightmost bin includes its right boundary
                if (pos == (num_bin_edges[dim] - 1)) {
                    pos -= 1;
                }

                hist_index += hist_strides[dim] * pos;
            }

            if (!skip_elt) {
                // In the unweighted case, the default weight is 1
                input_t wt = accessor_wt.has_value() ? accessor_wt.value()[i] : static_cast<input_t>(1);

                hist_local_data[hist_index] += wt;
            }
        }
    });

    at::sum_out(hist, thread_histograms, /*dim=*/{0});
}
```





## TensorFlow

无`histogramdd`实现

## Numpy

numpy.**histogramdd**(*sample*, *bins=10*, *range=None*, *density=None*, *weights=None*)[[source\]](https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/histograms.py#L901-L1072)

Compute the multidimensional histogram of some data.

- Parameters:

  **sample**: (N, D) array, or (N, D) array_likeThe data to be histogrammed.Note the unusual interpretation of sample when an array_like:When an array, each row is a coordinate in a D-dimensional space - such as `histogramdd(np.array([p1, p2, p3]))`.When an array_like, each element is the list of values for single coordinate - such as `histogramdd((X, Y, Z))`.The first form should be preferred.

  **bins**: sequence or int, optionalThe bin specification:A sequence of arrays describing the monotonically increasing bin edges along each dimension.The number of bins for each dimension (nx, ny, … =bins)The number of bins for all dimensions (nx=ny=…=bins).

  **range**: sequence, optionalA sequence of length D, each an optional (lower, upper) tuple giving the outer bin edges to be used if the edges are not given explicitly in *bins*. An entry of None in the sequence results in the minimum and maximum values being used for the corresponding dimension. The default, None, is equivalent to passing a tuple of D None values.

  **density**: bool, optionalIf False, the default, returns the number of samples in each bin. If True, returns the probability *density* function at the bin, `bin_count / sample_count / bin_volume`.

  **weights**: (N,) array_like, optionalAn array of values *w_i* weighing each sample *(x_i, y_i, z_i, …)*. Weights are normalized to 1 if density is True. If density is False, the values of the returned histogram are equal to the sum of the weights belonging to the samples falling into each bin.

- Returns:

  **H**: ndarrayThe multidimensional histogram of sample x. See density and weights for the different possible semantics.

  **edges**: listA list of D arrays describing the bin edges for each dimension.

### 实现方法

先模板生成函数，底层cpp调用实现[代码位置](https://github.com/numpy/numpy/blob/v1.26.0/numpy/lib/histograms.py#L901-L1072)

```python
@array_function_dispatch(_histogramdd_dispatcher)
def histogramdd(sample, bins=10, range=None, density=None, weights=None):
	try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, np.intp)
    edges = D*[None]
    dedges = D*[None]
    if weights is not None:
        weights = np.asarray(weights)

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                'sample x.')
    except TypeError:
        # bins is an integer
        bins = D*[bins]

    # normalize the range argument
    if range is None:
        range = (None,) * D
    elif len(range) != D:
        raise ValueError('range argument must have one entry per dimension')

    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:,i], range[i])
            try:
                n = operator.index(bins[i])

            except TypeError as e:
                raise TypeError(
                	"`bins[{}]` must be an integer, when a scalar".format(i)
                ) from e

            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                    .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = np.bincount(xy, weights, minlength=nbin.prod())

    # Shape into a proper matrix
    hist = hist.reshape(nbin)

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float, casting='safe')

    # Remove outliers (indices 0 and -1 for each dimension).
    core = D*(slice(1, -1),)
    hist = hist[core]

    if density:
        # calculate the probability density function
        s = hist.sum()
        for i in _range(D):
            shape = np.ones(D, int)
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s

    if (hist.shape != nbin - 2).any():
        raise RuntimeError(
            "Internal Shape Error")
    return hist, edges
```



# 四、对比分析

PyTorch底层用cpp实现kernel，Numpy通过API在Python层直接实现。

# 五、设计思路与实现方案

## 命名与参数设计

API的设计为:

- paddle.histogramdd(x, bins, ranges=None, density=False, weights=None，name=None)

其中

+ x(Tensor) - 输入的多维 tensor
+ bins(Tensor[], int[], int) 若为`Tensor[]`，则定义了bin的边缘序列；若为`int[]`，则每个值分别定义了每个维度的等宽bin的数量；若为`int`，则定义了所有维度的等宽bin的数量。
+ ranges(*sequence of python:float*)：规定了bin的最左端和最右端，也就是范围。若为None则以所有输入的最小值和最大值作为边界。
+ density (bool) – 默认为 False , 结果将包含每个bin中的计数。如果设置为 True ，则每个计数（重量）将除以总计数，然后除以其所在bin的范围宽度。
+ weight(Tensor): 默认所有输入权重为1，他的shape必须与输入sample除去最内部维度的shape相同，例如当sample的shape为[M,N]时，weight的shape必须为[M]。
+ name(str, 可选）- 操作的名称(默认值为None）。

## 底层OP设计

参考PyTorch与Numpy中的设计，通过组合Python API实现功能。

## API实现方案

1. 在 Paddle repo 的 python/paddle/linalg.py 文件中实现Python。
4. 编写文档
5. 编写单测

# 六、测试和验收的考量

单测代码位置，Paddle repo 的 test/legacy_test/test_histogramdd_op目录

测试考虑的case如下：

+ 验证正确性，与PyTorch中的API计算结果进行对比。
+ 修改参数，对不同参数的功能以及边界值进行测试。

# 七、可行性分析及规划排期

有业内方案实现作为参考，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[PyTorch文档](https://pytorch.org/docs/stable/generated/torch.histogramdd.html?highlight=histogramdd#torch.histogramdd)

[Numpy文档](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html#numpy-histogramdd)