# paddle.isposinf，paddle.isneginf，paddle.isreal，paddle.isin 设计文档

|API名称 | paddle.isposinf /paddle.isneginf /paddle.isreal /paddle.isin | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-03-22 | 
|版本号 | 此设计文档的版本号，如V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20240322_api_design_for_isposinf_isneginf_isreal_isin.md<br> | 


# 一、概述
## 1、相关背景
[NO.10 为 Paddle 新增 isposinf / isneginf / isreal / isin API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_6th/【Hackathon%206th】开源贡献个人挑战赛框架开发任务合集.md#no10-为-paddle-新增-isposinf--isneginf--isreal--isin-api)

## 2、功能目标
- 实现 paddle.isposinf 作为独立的函数调用，Tensor.isposinf(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为正无穷大。
- 实现 paddle.isneginf 作为独立的函数调用，Tensor.isneginf(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为负无穷大。
- 实现 paddle.isreal 作为独立的函数调用，Tensor.isreal(x) 做为 Tensor 的方法使用。测试 input 的每个元素是否为实值。
- 实现 paddle.isin 作为独立的函数调用，Tensor.isin(x) 做为 Tensor 的方法使用。测试 elements 的每个元素是否在 test_elements 中。

## 3、意义
新增 paddle.isposinf，paddle.isneginf，paddle.isreal，paddle.isin 方法，丰富 paddle API

# 二、飞桨现状
对于 paddle.isposinf，paddle.isneginf，paddle 目前有相似的API paddle.isinf；
对于 paddle.isreal 目前有相似的API paddle.is_complex；
对于 paddle.isin 暂无类似API。

# 三、业内方案调研

### PyTorch
PyTorch 中的 torch.isposinf API文档 (https://pytorch.org/docs/stable/generated/torch.isposinf.html#torch-isposinf)
PyTorch 中的 torch.isneginf API文档 (https://pytorch.org/docs/stable/generated/torch.isneginf.html#torch-isneginf)
PyTorch 中的 torch.isreal API文档 (https://pytorch.org/docs/stable/generated/torch.isreal.html#torch-isreal)
PyTorch 中的 torch.isin API文档 (https://pytorch.org/docs/stable/generated/torch.isin.html#torch-isin)

### Numpy
Numpy 中的 numpy.isposinf API文档 (https://numpy.org/doc/stable/reference/generated/numpy.isposinf.html)
Numpy 中的 numpy.isneginf API文档 (https://numpy.org/doc/stable/reference/generated/numpy.isneginf.html)
Numpy 中的 numpy.isreal API文档 (https://numpy.org/doc/stable/reference/generated/numpy.isreal.html)
Numpy 中的 numpy.isin API文档 (https://numpy.org/doc/stable/reference/generated/numpy.isin.html)

### 实现方法
- isposinf
    - pytorch
    ```cpp
    static void isposinf_kernel_impl(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isposinf_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t a) -> bool { return a == std::numeric_limits<scalar_t>::infinity(); });
    });
    }
    ```
    - numpy
    ```python
    def isposinf(x, out=None):
        is_inf = nx.isinf(x)
        try:
            signbit = ~nx.signbit(x)
        except TypeError as e:
            dtype = nx.asanyarray(x).dtype
            raise TypeError(f'This operation is not supported for {dtype} values '
                            'because it would be ambiguous.') from e
        else:
            return nx.logical_and(is_inf, signbit, out)
    ```

- isneginf
    - pytorch
    ```cpp
    static void isneginf_kernel_impl(TensorIteratorBase& iter) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.input_dtype(), "isneginf_cpu", [&]() {
        cpu_kernel(iter, [](scalar_t a) -> bool { return a == -std::numeric_limits<scalar_t>::infinity(); });
    });
    }
    ```
    - numpy
    ```python
    def isneginf(x, out=None):
        is_inf = nx.isinf(x)
        try:
            signbit = nx.signbit(x)
        except TypeError as e:
            dtype = nx.asanyarray(x).dtype
            raise TypeError(f'This operation is not supported for {dtype} values '
                            'because it would be ambiguous.') from e
        else:
            return nx.logical_and(is_inf, signbit, out)
    ```


- isreal
    - pytorch
    ```cpp
    Tensor isreal(const Tensor& self) {
    // Note: Integral and Floating tensor values are always real
    if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true) ||
        c10::isFloatingType(self.scalar_type())) {
        return at::ones_like(self, at::kBool, at::MemoryFormat::Preserve);
    }

    return at::imag(self) == 0;
    }
    ```
    - numpy
    ```python
    def isreal(x):
        return imag(x) == 0
    ```

- isin
pytorch
```cpp
// Default brute force implementation of isin(). Used when the number of test elements is small.
// Iterates through each element and checks it against each test element.
static void isin_default_kernel_cpu(
    const Tensor& elements,
    const Tensor& test_elements,
    bool invert,
    const Tensor& out) {
  // Since test elements is not an input of the TensorIterator, type promotion
  // must be done manually.
  ScalarType common_type = at::result_type(elements, test_elements);
  Tensor promoted_elements = elements.to(common_type);
  Tensor test_elements_flat = test_elements.to(common_type).view(-1);
  auto test_elements_stride = test_elements_flat.stride(0);

  auto iter = TensorIteratorConfig()
    .add_output(out)
    .add_const_input(promoted_elements)
    .check_all_same_dtype(false)
    .build();
  // Dispatch based on promoted type.
  AT_DISPATCH_ALL_TYPES(iter.dtype(1), "isin_default_cpu", [&]() {
    cpu_kernel(iter, [&](scalar_t element_val) -> bool {
      const auto* test_element_data = test_elements_flat.const_data_ptr<scalar_t>();
      for (const auto j : c10::irange(test_elements_flat.numel())) {
        if (element_val == *(test_element_data + test_elements_stride * j)) {
          return !invert;
        }
      }
      return invert;
    });
  });
}
```
```cpp
// Sorting-based algorithm for isin(); used when the number of test elements is large.
static void isin_sorting(
    const Tensor& elements,
    const Tensor& test_elements,
    bool assume_unique,
    bool invert,
    const Tensor& out) {
  // 1. Concatenate unique elements with unique test elements in 1D form. If
  //    assume_unique is true, skip calls to unique().
  Tensor elements_flat, test_elements_flat, unique_order;
  if (assume_unique) {
    elements_flat = elements.ravel();
    test_elements_flat = test_elements.ravel();
  } else {
    std::tie(elements_flat, unique_order) = at::_unique(
        elements, /*sorted=*/ false, /*return_inverse=*/ true);
    std::tie(test_elements_flat, std::ignore) = at::_unique(test_elements, /*sorted=*/ false);
  }

  // 2. Stable sort all elements, maintaining order indices to reverse the
  //    operation. Stable sort is necessary to keep elements before test
  //    elements within the sorted list.
  Tensor all_elements = at::cat({std::move(elements_flat), std::move(test_elements_flat)});
  auto [sorted_elements, sorted_order] = all_elements.sort(
      /*stable=*/ true, /*dim=*/ 0, /*descending=*/ false);

  // 3. Create a mask for locations of adjacent duplicate values within the
  //    sorted list. Duplicate values are in both elements and test elements.
  Tensor duplicate_mask = at::empty_like(sorted_elements, TensorOptions(ScalarType::Bool));
  Tensor sorted_except_first = sorted_elements.slice(0, 1, at::indexing::None);
  Tensor sorted_except_last = sorted_elements.slice(0, 0, -1);
  duplicate_mask.slice(0, 0, -1).copy_(
    invert ? sorted_except_first.ne(sorted_except_last) : sorted_except_first.eq(sorted_except_last));
  duplicate_mask.index_put_({-1}, invert);

  // 4. Reorder the mask to match the pre-sorted element order.
  Tensor mask = at::empty_like(duplicate_mask);
  mask.index_copy_(0, sorted_order, duplicate_mask);

  // 5. Index the mask to match the pre-unique element order. If
  //    assume_unique is true, just take the first N items of the mask,
  //    where N is the original number of elements.
  if (assume_unique) {
    out.copy_(mask.slice(0, 0, elements.numel()).view_as(out));
  } else {
    out.copy_(at::index(mask, {c10::optional<Tensor>(unique_order)}));
  }
}
```
```py
def isin(elements, test_elements, *, assume_unique=False, invert=False):
    # handle when either elements or test_elements are Scalars (they can't both be)
    if not isinstance(elements, torch.Tensor):
        elements = torch.tensor(elements, device=test_elements.device)
    if not isinstance(test_elements, torch.Tensor):
        test_elements = torch.tensor(test_elements, device=elements.device)

    if test_elements.numel() < 10.0 * pow(elements.numel(), 0.145):
        return isin_default(elements, test_elements, invert=invert)
    else:
        return isin_sorting(
            elements, test_elements, assume_unique=assume_unique, invert=invert
        )
```
```python
@register_decomposition(aten.isin)
@out_wrapper()
def isin(elements, test_elements, *, assume_unique=False, invert=False):
    # handle when either elements or test_elements are Scalars (they can't both be)
    if not isinstance(elements, torch.Tensor):
        elements = torch.tensor(elements, device=test_elements.device)
    if not isinstance(test_elements, torch.Tensor):
        test_elements = torch.tensor(test_elements, device=elements.device)

    if test_elements.numel() < 10.0 * pow(elements.numel(), 0.145):
        return isin_default(elements, test_elements, invert=invert)
    else:
        return isin_sorting(
            elements, test_elements, assume_unique=assume_unique, invert=invert
        )


def isin_default(elements, test_elements, *, invert=False):
    if elements.numel() == 0:
        return torch.empty_like(elements, dtype=torch.bool)

    x = elements.view(*elements.shape, *((1,) * test_elements.ndim))
    if not invert:
        cmp = x == test_elements
    else:
        cmp = x != test_elements
    dim = tuple(range(-1, -test_elements.ndim - 1, -1))
    return cmp.any(dim=dim)


def isin_sorting(elements, test_elements, *, assume_unique=False, invert=False):
    elements_flat = elements.flatten()
    test_elements_flat = test_elements.flatten()
    if assume_unique:
        # This is the same as the aten implementation. For
        # assume_unique=False, we cannot use unique() here, so we use a
        # version with searchsorted instead.
        all_elements = torch.cat([elements_flat, test_elements_flat])
        sorted_elements, sorted_order = torch.sort(all_elements, stable=True)

        duplicate_mask = sorted_elements[1:] == sorted_elements[:-1]
        duplicate_mask = torch.constant_pad_nd(duplicate_mask, [0, 1], False)

        if invert:
            duplicate_mask = duplicate_mask.logical_not()

        mask = torch.empty_like(duplicate_mask)
        mask = mask.index_copy(0, sorted_order, duplicate_mask)

        return mask[0 : elements.numel()]
    else:
        sorted_test_elements, _ = torch.sort(test_elements_flat)
        idx = torch.searchsorted(sorted_test_elements, elements_flat)
        test_idx = torch.where(idx < sorted_test_elements.numel(), idx, 0)
        cmp = sorted_test_elements[test_idx] == elements_flat
        cmp = cmp.logical_not() if invert else cmp
        return cmp.reshape(elements.shape)
```
numpy
```python
def isin(element, test_elements, assume_unique=False, invert=False, *,
         kind=None):
    element = np.asarray(element)
    return in1d(element, test_elements, assume_unique=assume_unique,
                invert=invert, kind=kind).reshape(element.shape)

def in1d(ar1, ar2, assume_unique=False, invert=False, *, kind=None):
    # Ravel both arrays, behavior for the first array could be different
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # Ensure that iteration through object arrays yields size-1 arrays
    if ar2.dtype == object:
        ar2 = ar2.reshape(-1, 1)

    if kind not in {None, 'sort', 'table'}:
        raise ValueError(
            f"Invalid kind: '{kind}'. Please use None, 'sort' or 'table'.")

    # Can use the table method if all arrays are integers or boolean:
    is_int_arrays = all(ar.dtype.kind in ("u", "i", "b") for ar in (ar1, ar2))
    use_table_method = is_int_arrays and kind in {None, 'table'}

    if use_table_method:
        if ar2.size == 0:
            if invert:
                return np.ones_like(ar1, dtype=bool)
            else:
                return np.zeros_like(ar1, dtype=bool)

        # Convert booleans to uint8 so we can use the fast integer algorithm
        if ar1.dtype == bool:
            ar1 = ar1.astype(np.uint8)
        if ar2.dtype == bool:
            ar2 = ar2.astype(np.uint8)

        ar2_min = np.min(ar2)
        ar2_max = np.max(ar2)

        ar2_range = int(ar2_max) - int(ar2_min)

        # Constraints on whether we can actually use the table method:
        #  1. Assert memory usage is not too large
        below_memory_constraint = ar2_range <= 6 * (ar1.size + ar2.size)
        #  2. Check overflows for (ar2 - ar2_min); dtype=ar2.dtype
        range_safe_from_overflow = ar2_range <= np.iinfo(ar2.dtype).max
        #  3. Check overflows for (ar1 - ar2_min); dtype=ar1.dtype
        if ar1.size > 0:
            ar1_min = np.min(ar1)
            ar1_max = np.max(ar1)

            # After masking, the range of ar1 is guaranteed to be
            # within the range of ar2:
            ar1_upper = min(int(ar1_max), int(ar2_max))
            ar1_lower = max(int(ar1_min), int(ar2_min))

            range_safe_from_overflow &= all((
                ar1_upper - int(ar2_min) <= np.iinfo(ar1.dtype).max,
                ar1_lower - int(ar2_min) >= np.iinfo(ar1.dtype).min
            ))

        # Optimal performance is for approximately
        # log10(size) > (log10(range) - 2.27) / 0.927.
        # However, here we set the requirement that by default
        # the intermediate array can only be 6x
        # the combined memory allocation of the original
        # arrays. See discussion on 
        # https://github.com/numpy/numpy/pull/12065.

        if (
            range_safe_from_overflow and 
            (below_memory_constraint or kind == 'table')
        ):

            if invert:
                outgoing_array = np.ones_like(ar1, dtype=bool)
            else:
                outgoing_array = np.zeros_like(ar1, dtype=bool)

            # Make elements 1 where the integer exists in ar2
            if invert:
                isin_helper_ar = np.ones(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 0
            else:
                isin_helper_ar = np.zeros(ar2_range + 1, dtype=bool)
                isin_helper_ar[ar2 - ar2_min] = 1

            # Mask out elements we know won't work
            basic_mask = (ar1 <= ar2_max) & (ar1 >= ar2_min)
            outgoing_array[basic_mask] = isin_helper_ar[ar1[basic_mask] -
                                                        ar2_min]

            return outgoing_array
        elif kind == 'table':  # not range_safe_from_overflow
            raise RuntimeError(
                "You have specified kind='table', "
                "but the range of values in `ar2` or `ar1` exceed the "
                "maximum integer of the datatype. "
                "Please set `kind` to None or 'sort'."
            )
    elif kind == 'table':
        raise ValueError(
            "The 'table' method is only "
            "supported for boolean or integer arrays. "
            "Please select 'sort' or None for kind."
        )


    # Check if one of the arrays may contain arbitrary objects
    contains_object = ar1.dtype.hasobject or ar2.dtype.hasobject

    # This code is run when
    # a) the first condition is true, making the code significantly faster
    # b) the second condition is true (i.e. `ar1` or `ar2` may contain
    #    arbitrary objects), since then sorting is not guaranteed to work
    if len(ar2) < 10 * len(ar1) ** 0.145 or contains_object:
        if invert:
            mask = np.ones(len(ar1), dtype=bool)
            for a in ar2:
                mask &= (ar1 != a)
        else:
            mask = np.zeros(len(ar1), dtype=bool)
            for a in ar2:
                mask |= (ar1 == a)
        return mask

    # Otherwise use sorting
    if not assume_unique:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)

    ar = np.concatenate((ar1, ar2))
    # We need this to be a stable sort, so always use 'mergesort'
    # here. The values from the first array should always come before
    # the values from the second array.
    order = ar.argsort(kind='mergesort')
    sar = ar[order]
    if invert:
        bool_ar = (sar[1:] != sar[:-1])
    else:
        bool_ar = (sar[1:] == sar[:-1])
    flag = np.concatenate((bool_ar, [invert]))
    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = flag

    if assume_unique:
        return ret[:len(ar1)]
    else:
        return ret[rev_idx]
```

# 四、对比分析
Numpy 中 isposinf，isneginf，isreal 的实现比较直接。
Numpy 的 isin 除了基于排序的算法，还针对int和bool输入类型实现了'table'算法，但有一定的内存使用局限性。
PyTorch 与 Numpy 的 isin 基于排序的算法实现相似，分为两种情况，当test_elements元素个数较少时直接进行暴力搜索，较多时则采取排序算法，可通过 Paddle 现有的 API 组合实现。

# 五、设计思路与实现方案

## 命名与参数设计
API `paddle.isposinf(x, name)`
paddle.isposinf
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
- name  (str, optional) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
:::::::::
- Tensor 返回输入 `x` 的每一个元素是否为 `+INF` 。

API `paddle.isneginf(x, name)`
paddle.isneginf
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
- name  (str, optional) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
:::::::::
- Tensor 返回输入 `x` 的每一个元素是否为 `-INF` 。

API `paddle.isreal(x, name)`
paddle.isreal
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
- name  (str, optional) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
:::::::::
- Tensor 返回输入 `x` 的每一个元素是否为实数。

API `paddle.isin(elements, test_elements, invert=False, name)`
paddle.isin
----------------------
参数
:::::::::
- elements (Tensor) - 输入 Tensor。
- test_elements (Tensor) - 用于检验每个 `elements` 元素的 Tensor。
- invert (bool, optional) - 若为 True ，返回值将逐元素取反。默认值为 False。
- name  (str, optional) - 具体用法请参见 [Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_guides/low_level/program.html#api-guide-name)，一般无需设置，默认值为 None。
:::::::::
- Tensor 返回 `elements` 的每一个元素是否在 `test_elements` 中，输出形状与 `elements` 一致。

## 底层OP设计
用现有API组合实现

## API实现方案
1. paddle.isposinf
利用 paddle.isinf 与 paddle.signbit 组合实现

2. paddle.isneginf
利用 paddle.isinf 与 paddle.signbit 组合实现

3. paddle.isreal
利用Tensor数据类型判断和 paddle.imag 实现

4. paddle.isin
参考 pytorch 在 _decompose 中的设计：当test_elements元素个数较少时直接进行暴力搜索，较多时则采取基于排序的算法（利用 flatten，concat，index_put_，searchsorted等API组合实现）。暂时去掉 assume_unique 参数，因为当前 paddle 的 argsort kernel 使用的是 std::sort 的不稳定排序，与 pytorch 和 numpy 的结果就会存在差异。若后期需要加 assume_unique 参数并用 argsort 实现 isin，则需要先实现 stable 的 argsort。

# 六、测试和验收的考量

测试case：

paddle.isposinf，paddle.isneginf，paddle.isin：
- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算；
  - 计算dtype类型：验证 `float64`，`int32`等；
- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：输入类型异常。

paddle.isreal：
- 正确性验证：可以与 NumPy 的结果对齐；
  - 不同 shape；
  - 前向计算；
  - 计算dtype类型：验证 `float64`，`int32`，`complex64`等；
- 不同计算设备：覆盖 CPU 和 GPU 等实现；
- 错误检查：输入类型异常。

# 七、可行性分析和排期规划

2024/03/24 - 2024/03/31 完成 API 主体实现；
2024/03/31 - 2024/04/07 完成单测；

# 八、影响面
丰富 paddle API，对其他模块没有影响
