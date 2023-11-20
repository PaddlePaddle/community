# paddle.pdist设计文档

| API 名称     | paddle.pdist                  |
| ------------ | ----------------------------- |
| 提交作者     | coco                          |
| 提交时间     | 2023-09-26                    |
| 版本号       | V1.0                          |
| 依赖飞桨版本 | develop                       |
| 文件名       | 20230926_api_defign_for_pdist |

# 一、概述

## 1、相关背景

为paddle新增该API，为计算N个向量两两之间的p-norm距离。

## 2、功能目标

一个矩阵`A`的大小为`MxN`，那么`B=pdist(A)`得到的矩阵B的大小为1行`M*(M-1)/2`列，表示的意义是M行数据，每两行计算一下p-norm距离，默认欧式距离。例如a = [[0.0, 1.0],[2.0,3.0],[4.0,5.0],[6.0,7.0]]，输出为[2.8284, 5.6569, 8.4853, 2.8284, 5.6569, 2.8284]。输出顺序为distance(第一行,第二行), distance(第一行,第三行), ... distance(第二行,第三行)...

## 3、意义

飞桨支持计算大小为(NxM)的矩阵中，N个向量两两之间的p-norm距离。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Scipy

Scipy中有API`scipy.spatial.distance.pdist`

在Scipy中介绍为：

```
Pairwise distances between observations in n-dimensional space.
```

## 实现方法

从实现方法上，Scipy是通过py实现的，[代码位置](https://github.com/scipy/scipy/blob/v1.11.2/scipy/spatial/distance.py#L2195-L2233)

```python
    X = _asarray_validated(X, sparse_ok=False, objects_ok=True, mask_ok=True,
                           check_finite=False)

    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m, n = s

    if callable(metric):
        mstr = getattr(metric, '__name__', 'UnknownCustomMetric')
        metric_info = _METRIC_ALIAS.get(mstr, None)

        if metric_info is not None:
            X, typ, kwargs = _validate_pdist_input(
                X, m, n, metric_info, **kwargs)

        return _pdist_callable(X, metric=metric, out=out, **kwargs)
    elif isinstance(metric, str):
        mstr = metric.lower()
        metric_info = _METRIC_ALIAS.get(mstr, None)

        if metric_info is not None:
            pdist_fn = metric_info.pdist_func
            _extra_windows_error_checks(X, out, (m * (m - 1) / 2,), **kwargs)
            return pdist_fn(X, out=out, **kwargs)
        elif mstr.startswith("test_"):
            metric_info = _TEST_METRICS.get(mstr, None)
            if metric_info is None:
                raise ValueError(f'Unknown "Test" Distance Metric: {mstr[5:]}')
            X, typ, kwargs = _validate_pdist_input(
                X, m, n, metric_info, **kwargs)
            return _pdist_callable(
                X, metric=metric_info.dist_func, out=out, **kwargs)
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
```

先找到`mertric`对应的函数，然后call调用，例如`metric`为`euclidean`时，调用`euclidean`的函数。[代码位置](https://github.com/scipy/scipy/blob/v1.11.2/scipy/spatial/distance.py#L1781C1-L1787C7)



```python
    MetricInfo(
        canonical_name='euclidean',
        aka={'euclidean', 'euclid', 'eu', 'e'},
        dist_func=euclidean,
        cdist_func=_distance_pybind.cdist_euclidean,
        pdist_func=_distance_pybind.pdist_euclidean,
    ),
```

[euclidean调用minkowski](https://github.com/scipy/scipy/blob/v1.11.2/scipy/spatial/distance.py#L500-L536)和[minkowski实现](https://github.com/scipy/scipy/blob/v1.11.2/scipy/spatial/distance.py#L429-L497)

```python
def euclidean(u, v, w=None):
    return minkowski(u, v, p=2, w=w)


def minkowski(u, v, p=2, w=None):
    u = _validate_vector(u)
    v = _validate_vector(v)
    if p <= 0:
        raise ValueError("p must be greater than 0")
    u_v = u - v
    if w is not None:
        w = _validate_weights(w)
        if p == 1:
            root_w = w
        elif p == 2:
            # better precision and speed
            root_w = np.sqrt(w)
        elif p == np.inf:
            root_w = (w != 0)
        else:
            root_w = np.power(w, 1/p)
        u_v = root_w * u_v
    dist = norm(u_v, ord=p)
    return dist
```

主要是调用`norm`实现计算

```python
def norm(x, ord=None, axis=None):
    if not issparse(x):
        raise TypeError("input is not sparse. use numpy.linalg.norm")

    # Check the default case first and handle it immediately.
    if axis is None and ord in (None, 'fro', 'f'):
        return _sparse_frobenius_norm(x)

    # Some norms require functions that are not implemented for all types.
    x = x.tocsr()

    if axis is None:
        axis = (0, 1)
    elif not isinstance(axis, tuple):
        msg = "'axis' must be None, an integer or a tuple of integers"
        try:
            int_axis = int(axis)
        except TypeError as e:
            raise TypeError(msg) from e
        if axis != int_axis:
            raise TypeError(msg)
        axis = (int_axis,)

    nd = 2
    if len(axis) == 2:
        row_axis, col_axis = axis
        if not (-nd <= row_axis < nd and -nd <= col_axis < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' %
                             (axis, x.shape))
        if row_axis % nd == col_axis % nd:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            # Only solver="lobpcg" supports all numpy dtypes
            _, s, _ = svds(x, k=1, solver="lobpcg")
            return s[0]
        elif ord == -2:
            raise NotImplementedError
            #return _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            return abs(x).sum(axis=row_axis).max(axis=col_axis)[0,0]
        elif ord == np.inf:
            return abs(x).sum(axis=col_axis).max(axis=row_axis)[0,0]
        elif ord == -1:
            return abs(x).sum(axis=row_axis).min(axis=col_axis)[0,0]
        elif ord == -np.inf:
            return abs(x).sum(axis=col_axis).min(axis=row_axis)[0,0]
        elif ord in (None, 'f', 'fro'):
            # The axis order does not matter for this norm.
            return _sparse_frobenius_norm(x)
        else:
            raise ValueError("Invalid norm order for matrices.")
    elif len(axis) == 1:
        a, = axis
        if not (-nd <= a < nd):
            raise ValueError('Invalid axis %r for an array with shape %r' %
                             (axis, x.shape))
        if ord == np.inf:
            M = abs(x).max(axis=a)
        elif ord == -np.inf:
            M = abs(x).min(axis=a)
        elif ord == 0:
            # Zero norm
            M = (x != 0).sum(axis=a)
        elif ord == 1:
            # special case for speedup
            M = abs(x).sum(axis=a)
        elif ord in (2, None):
            M = sqrt(abs(x).power(2).sum(axis=a))
        else:
            try:
                ord + 1
            except TypeError as e:
                raise ValueError('Invalid norm order for vectors.') from e
            M = np.power(abs(x).power(ord).sum(axis=a), 1 / ord)
        if hasattr(M, 'toarray'):
            return M.toarray().ravel()
        elif hasattr(M, 'A'):
            return M.A.ravel()
        else:
            return M.ravel()
    else:
        raise ValueError("Improper number of dimensions to norm.")
```









## PyTorch

Parameters:

- **input** – input tensor of shape N×M.
- **p** – p value for the p-norm distance to calculate between each vector pair ∈[0,∞]∈[0,∞].

并且有相关描述：

This function is equivalent to `scipy.spatial.distance.pdist(input, 'minkowski', p=p)` if p∈(0,∞). When p=0 it is equivalent to `scipy.spatial.distance.pdist(input, 'hamming') * M`. When p=∞, the closest scipy function is `scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`.



相关[实现位置](https://github.com/pytorch/pytorch/blob/d0f82cd082fad7243226e0ab68fd995873ea7d76/aten/src/ATen/native/Distance.cpp#L58-L64)

```cpp
Tensor pdist(const Tensor& self, const double p) {
  TORCH_CHECK(self.dim() == 2,
      "pdist only supports 2D tensors, got: ", self.dim(), "D");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "pdist only supports floating-point dtypes");
  TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");
  return at::_pdist_forward(self.contiguous(), p);
}
```

调用`_pdist_forward`，[实现位置](https://github.com/pytorch/pytorch/blob/d0f82cd082fad7243226e0ab68fd995873ea7d76/aten/src/ATen/native/Distance.cpp#L244-L262)

```cpp
Tensor _pdist_forward(const Tensor& self, const double p) {
  TORCH_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
  auto device = self.device().type();
  TORCH_CHECK(device == kCPU || device == kCUDA, "_pdist_forward only supports CPU and CUDA devices, got: ", device);
  Tensor result = at::empty({0}, self.options(), LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  if (self.size(0) <= 1) {
    result.resize_({0});
  } else {
    int64_t n = self.size(0);
    int64_t c = n * (n - 1) / 2;
    result.resize_({c});
    if (self.size(1) == 0) {
      result.fill_(0);
    } else {
      pdist_forward_stub(device, result, self, p);
    }
  }
  return result;
}
```

主要调用`pdist_forward_stub`，绑定了具体的`pdist_forward_kernel_impl`

```cpp
REGISTER_DISPATCH(pdist_forward_stub, &pdist_forward_kernel_impl);
```

([CPU](https://github.com/pytorch/pytorch/blob/d0f82cd082fad7243226e0ab68fd995873ea7d76/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp#L446)和[CUDA](https://github.com/pytorch/pytorch/blob/d0f82cd082fad7243226e0ab68fd995873ea7d76/aten/src/ATen/native/cuda/DistanceKernel.cu#L360)实现绑定了同一个`pdist_forward_kernel_impl`)

而后`pdist_forward_kernel_impl`的[实现位置](https://github.com/pytorch/pytorch/blob/d0f82cd082fad7243226e0ab68fd995873ea7d76/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp#L419C1-L423C2)

```cpp
void pdist_forward_kernel_impl(Tensor& result, const Tensor& self, const double p) {
  AT_DISPATCH_FLOATING_TYPES(self.scalar_type(), "pdist", [&] {
    Dist<scalar_t>::apply_pdist(result, self, p);
  });
}
```

调用`apply_pdist`，[代码位置](https://github.com/pytorch/pytorch/blob/d0f82cd082fad7243226e0ab68fd995873ea7d76/aten/src/ATen/native/cpu/DistanceOpsKernel.cpp#L190-L202)

```cpp
 // Assumes self is nonempty, contiguous, and 2D
  static void apply_pdist(Tensor& result, const Tensor& self, const scalar_t p) {
    if (p == 0.0) {
      run_parallel_pdist<zdist_calc<Vec>>(result, self, p);
    } else if (p == 1.0) {
      run_parallel_pdist<odist_calc<Vec>>(result, self, p);
    } else if (p == 2.0) {
      run_parallel_pdist<tdist_calc<Vec>>(result, self, p);
    } else if (std::isinf(p)) {
      run_parallel_pdist<idist_calc<Vec>>(result, self, p);
    } else {
      run_parallel_pdist<pdist_calc<Vec>>(result, self, p);
    }
  }
```

`run_parallel_pdist`具体实现

```cpp
  template <typename F>
  static void run_parallel_pdist(Tensor& result, const Tensor& self, const scalar_t p) {
    const scalar_t * const self_start = self.data_ptr<scalar_t>();
    const scalar_t * const self_end = self_start + self.numel();
    int64_t n = self.size(0);
    int64_t m = self.size(1);

    scalar_t * const res_start = result.data_ptr<scalar_t>();
    int64_t combs = result.numel(); // n * (n - 1) / 2

    // We conceptually iterate over tuples of (i, j, k) where i is the first
    // vector from the input, j is the second, and k is the result index. This
    // parallelizes over the range of k and infers what i and j are from the
    // value of k.
    parallel_for(0, combs, internal::GRAIN_SIZE / (16 * m), [p, self_start, self_end, n, m, res_start](int64_t k, int64_t end) {
      const Vec pvec(p);
      double n2 = n - .5;
      // The -1 accounts for floating point truncation issues
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      int64_t i = static_cast<int64_t>((n2 - std::sqrt(n2 * n2 - 2 * k - 1)));
      int64_t j = k - n * i + i * (i + 1) / 2 + i + 1;

      const scalar_t * self_i = self_start + i * m;
      const scalar_t * self_j = self_start + j * m;
      scalar_t * res = res_start + k;
      const scalar_t * const res_end = res_start + end;

      while (res != res_end) {
        *res = F::finish(vec::map2_reduce_all<scalar_t>(
          [&pvec](Vec a, Vec b) { return F::map((a - b).abs(), pvec); },
          F::red, self_i, self_j, m), p);

        res += 1;
        self_j += m;
        if (self_j == self_end) {
          self_i += m;
          self_j = self_i + m;
        }
      }
    });
  }
```



# 四、对比分析

Scipy利用现有API组合实现，PyTorch则在底层重写cpp算子。

# 五、设计思路与实现方案

## 命名与参数设计

API的设计为:

`paddle.pdist(x, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary", name=None)`

Args：

+ x(Tensor): 严格为 shape=[M, N] 的 Tensor
+ p(float, 可选): 为p-范数对应的p值，默认为2.0
+ compute_mode(str, 可选): 默认为`use_mm_for_euclid_dist_if_necessary`（组合已有API过程中用到了`paddle.cdist`，当`p=2.0`时，可以设置`compute_mode`利用矩阵运算进行优化）
  + `compute_mode=use_mm_for_euclid_dist_if_necessary`时，当p=2.0且M>25时使用矩阵乘法计算距离
  + `compute_mode=use_mm_for_euclid_dist`时，当p=2.0时使用矩阵乘法计算距离
  + `compute_mode=donot_use_mm_for_euclid_dist`时，不使用矩阵乘法计算距离
+ name(str, 可选): 操作的名称(默认为None)

Return：

+ 一行 `Mx(M-1)/2` 列的 Tensor



## API实现方案

参考`PyTorch`与`Scipy`中的设计，组合已有API实现功能：

在 Paddle repo 的 ﻿python/paddle/nn/functional/distance.py文件；并在 ﻿python/paddle/nn/functional/init.py中，添加 pdist API，以支持 paddle.Tensor.pdist 的调用方式；

使用的API：`paddle.cdist`,`paddle.tril`,`paddle.masked_select`

# 六、测试和验收的考量

单测代码位置，Paddle repo 的 paddle/test/legacy_test/test_pdist.py 目录

测试考虑的case如下：

1. 当`x`、`y` 2D 的 Tensor，并如PyTorch给出合理提示

   ```python
   >>> a = []
   >>> a = torch.tensor(a)
   >>> b = torch.nn.functional.pdist(a)
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   RuntimeError: pdist only supports 2D tensors, got: 1D
   >>> b
   ```

   

2. 结果一致性，和 SciPy 以及 PyTorch 结果的数值的一致性

# 七、可行性分析及规划排期

有业内方案实现作为参考，工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

[PyTorch文档](https://pytorch.org/docs/stable/generated/torch.nn.functional.pdist.html?highlight=pdist#torch.nn.functional.pdist)

[Scipy文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)