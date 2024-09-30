# paddle.Tensor.set_，paddle.Tensor.resize_ 设计文档

|API名称 | paddle.Tensor.set_ /paddle.Tensor.resize_ | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-09-26 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20240926_api_design_for_set__resize_.md<br> | 


# 一、概述
## 1、相关背景
[NO.20 为 Paddle 新增 Tensor.set_ / Tensor.resize_ API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/【Hackathon%207th】个人挑战赛—框架开发任务合集.md#no20-为-paddle-新增-tensorset--tensorresize-api)

## 2、功能目标
- 实现 paddle.Tensor.set_ 做为 Tensor 的方法使用。设置 Tensor 与 source 共享相同的存储空间，且可设置为与 source 相同或不同的 shape，stride，offset。
- 实现 paddle.Tensor.resize_ 做为 Tensor 的方法使用。重新调整 Tensor 的 size，若新 numel 小于等于原 numel，不改变存储空间大小；若新 numel 大于原 numel，需调整存储空间大小。

## 3、意义
新增 paddle.Tensor.set_，paddle.Tensor.resize_ 方法，丰富 paddle API

# 二、飞桨现状
对于 paddle.Tensor.set_，paddle 目前有相似的 API paddle.as_strided；
对于 paddle.resize_ 目前无相似实现，可以在实现 paddle.Tensor.set_ 后使用 set_ 组合开发；

# 三、业内方案调研

### PyTorch
- PyTorch 中的 torch.Tensor.set_ [API文档](https://pytorch.org/docs/stable/generated/torch.Tensor.set_.html)
- PyTorch 中的 torch.Tensor.resize_ [API文档](https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html)

### Numpy
- Numpy 中的 numpy.resize [API文档](https://numpy.org/doc/stable/reference/generated/numpy.resize.html)

### 实现方法
- Tensor.set_
    - pytorch
    ```cpp
    static PyObject* THPVariable_set_(
                                    PyObject* self_,
                                    PyObject* args,
                                    PyObject* kwargs) {
        HANDLE_TH_ERRORS
        const Tensor& self = THPVariable_Unpack(self_);
        static PythonArgParser parser(
            {
                "set_()",
                "set_(Storage source)",
                "set_(Storage source, SymInt storage_offset, SymIntArrayRef size, SymIntArrayRef stride=None)",
                "set_(Tensor source)",
                "set_(Tensor source, SymInt storage_offset, SymIntArrayRef size, SymIntArrayRef stride=None)",
            },
            /*traceable=*/false);

        ParsedArgs<4> parsed_args;
        auto _r = parser.parse(args, kwargs, parsed_args);

        switch (_r.idx) {
            case 0: {
            // aten::set_(Tensor(a!) self) -> Tensor(a!)
            auto dispatch_set_ = [](const Tensor& self) -> Tensor {
                pybind11::gil_scoped_release no_gil;
                return self.set_();
            };
            return wrap(dispatch_set_(self));
            }
            case 1: {
            // aten::set_.source_Storage(Tensor(a!) self, Storage source) ->
            // Tensor(a!)
            at::ScalarType storage_scalar_type;
            bool is_typed_storage = true;
            at::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
            TORCH_CHECK(storage_scalar_type == self.dtype() || !is_typed_storage,
                "Expected a Storage of type ", self.dtype(),
                " or an UntypedStorage, but got type ", storage_scalar_type,
                " for argument 1 'storage'");
            auto dispatch_set_ = [](const Tensor& self, Storage source) -> Tensor {
                pybind11::gil_scoped_release no_gil;
                return self.set_(source);
            };
            return wrap(dispatch_set_(self, storage));
            }
            case 2: {
            // aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage
            // source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
            at::ScalarType storage_scalar_type;
            bool is_typed_storage = true;
            at::Storage storage = _r.storage(0, storage_scalar_type, is_typed_storage);
            TORCH_CHECK(storage_scalar_type == self.dtype() || !is_typed_storage,
                "Expected a Storage of type ", self.dtype(),
                " or an UntypedStorage, but got type ", storage_scalar_type,
                " for argument 1 'storage'");
            auto dispatch_set_ = [](const Tensor& self,
                                    Storage source,
                                    c10::SymInt storage_offset,
                                    c10::SymIntArrayRef size,
                                    c10::SymIntArrayRef stride) -> Tensor {
                pybind11::gil_scoped_release no_gil;
                return self.set__symint(source, storage_offset, size, stride);
            };
            return wrap(dispatch_set_(
                self, storage, _r.toSymInt(1), _r.symintlist(2), _r.symintlist(3)));
            }
            case 3: {
            // aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
            auto dispatch_set_ = [](const Tensor& self, const Tensor& source) -> Tensor {
                TORCH_CHECK(source.dtype() == self.dtype(), "Could not set tensor of type ", source.dtype(), " to a tensor of type ", self.dtype());
                pybind11::gil_scoped_release no_gil;
                return self.set_(source);
            };
            return wrap(dispatch_set_(self, _r.tensor(0)));
            }
            case 4: {
            // aten::set_.source_Tensor_storage_offset(Tensor(a!) self, Tensor
            // source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
            at::Tensor storage = _r.tensor(0);
            auto dispatch_set_ = [](const Tensor& self,
                                    const Tensor& source,
                                    c10::SymInt storage_offset,
                                    c10::SymIntArrayRef size,
                                    c10::SymIntArrayRef stride) -> Tensor {
                pybind11::gil_scoped_release no_gil;
                return self.set__symint(source, storage_offset, size, stride);
            };
            return wrap(dispatch_set_(
                self, storage, _r.toSymInt(1), _r.symintlist(2), _r.symintlist(3)));
            }
        }
        Py_RETURN_NONE;
        END_HANDLE_TH_ERRORS
    }


    {"set_", castPyCFunctionWithKeywords(THPVariable_set_), METH_VARARGS | METH_KEYWORDS, NULL},


    void FunctionalTensorWrapper::set__impl(const FunctionalTensorWrapper* other) {
        // self.set_(src) will cause self to have all of the tensor properties of self.
        value_ = other->value_;
        generation_ = other->generation_;
        view_metas_ = other->view_metas_;
        is_symbolic_ = other->is_symbolic_;
        // FREEZE the old storage, preventing mutations to it.
        // this is a huge pain to handle properly in all cases, so we ban it.
        functional_storage_impl()->freeze();
        // Unsafely swap out the storage with other's storage,
        // disconnecting `self` with its view chain
        storage_ = other->storage_;
        /// explicitly mark the tensor as having its storage changed from set_()
        // Otherwise, we don't actually have a 100% accurate way to check this.
        // (We could check if the updated value has a new storage than the original value,
        // but this won't also let us uniquely determine if the tensor **also**
        // experienced a data mutation).
        was_storage_changed_ = true;

        auto sizes_ = value_.sym_sizes();
        auto strides_ = value_.sym_strides();
        auto storage_offset_ = value_.sym_storage_offset();
        set_sizes_and_strides(sizes_, strides_, storage_offset_);
    }

    void TensorImpl::set_sizes_and_strides(
    c10::SymIntArrayRef sizes,
    c10::SymIntArrayRef strides,
    std::optional<c10::SymInt> storage_offset) {
        auto int_sizes = asIntArrayRefSlowOpt(sizes);
        auto int_strides = asIntArrayRefSlowOpt(strides);
        if (int_sizes && int_strides &&
            // NB: storage_offset guaranteed to be positive
            (!storage_offset.has_value() || !storage_offset->is_heap_allocated()) &&
            !has_symbolic_sizes_strides_) {
            set_sizes_and_strides(*int_sizes, *int_strides);
            if (storage_offset.has_value())
            set_storage_offset(storage_offset->as_int_unchecked());
            return;
        }
        TORCH_CHECK(
            allow_tensor_metadata_change(),
            "set_sizes_and_strides ",
            err_msg_tensor_metadata_change_not_allowed);

        has_symbolic_sizes_strides_ = true;
        refresh_sizes_strides_policy();
        if (!extra_meta_) {
            extra_meta_ = std::make_unique<ExtraMeta>();
            extra_meta_->symbolic_shape_meta_ =
                std::make_unique<c10::SymbolicShapeMeta>();
            extra_meta_->symbolic_shape_meta_->strides_valid_ = !is_sparse();
            if (!storage_offset.has_value()) {
            extra_meta_->symbolic_shape_meta_->storage_offset_ = storage_offset_;
            }
        }

        auto& sym_shape_meta{symbolic_shape_meta()};
        clone_symvec(sizes, sym_shape_meta.sizes_);
        clone_symvec(strides, sym_shape_meta.strides_);
        if (storage_offset.has_value())
            sym_shape_meta.storage_offset_ = storage_offset->clone();

        refresh_numel();
        refresh_contiguous();
    }
    ```

- Tensor.resize_
    - pytorch
    ```cpp
    const Tensor& resize_(
            const Tensor& self,
            IntArrayRef size,
            std::optional<MemoryFormat> optional_memory_format) {
        if (self.has_names()) {
            return resize_named_tensor_(self, size, optional_memory_format);
        }
        return _resize_(self, size, optional_memory_format);
    }

    template <typename T>
    const Tensor& _resize_(
        const Tensor& self,
        ArrayRef<T> size,
        std::optional<MemoryFormat> optional_memory_format) {
        auto* self_ = self.unsafeGetTensorImpl();
        int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().sym_nbytes().maybe_as_int().value_or(-1) : 0;
        // NOLINTNEXTLINE(bugprone-argument-comment)
        _resize_impl_<T>(self_, size, /*strides=*/std::nullopt, true);
        if (optional_memory_format.has_value()) {
            auto memory_format =
                optional_memory_format.value();
            TORCH_CHECK(
                memory_format != MemoryFormat::Preserve,
                "Unsupported memory format",
                memory_format);
            self_->empty_tensor_restride(memory_format);
        }
        // See Note [Enabling Deterministic Operations]
        if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory() && old_storage_nbytes != -1)) {
            at::native::fill_resize_deterministic_(self, old_storage_nbytes);
        }
        return self;
    }

    template <typename T>
    TensorImpl* _resize_impl_(
        TensorImpl* self,
        ArrayRef<T> size,
        at::OptionalArrayRef<T> stride,
        bool resize_storage) {
        if (self->generic_sizes<T>() == size && (!stride || self->generic_strides<T>() == stride.value())) {
            return self;
        }

        const auto itemsize = self->dtype().itemsize();
        const auto storage_offset = self->generic_storage_offset<T>();
        T storage_size = T(1);
        if (stride) {
            self->set_sizes_and_strides(size, *stride);
            storage_size = at::detail::computeStorageNbytes(
                size, *stride, itemsize, storage_offset);
        } else {
            self->generic_set_sizes_contiguous(size);
            storage_size = at::detail::computeStorageNbytesContiguous(
                size, itemsize, storage_offset);
        }

        if (resize_storage) {
            _maybe_resize_storage(self, std::move(storage_size));
        }

        return self;
    }

    static void _maybe_resize_storage(TensorImpl* self, int64_t new_size_bytes) {
        maybe_resize_storage_cpu(self, new_size_bytes);
    }

    static void _maybe_resize_storage(TensorImpl* self, c10::SymInt new_size_bytes) {
        if (self->is_cpu()) {
            maybe_resize_storage_cpu(self, new_size_bytes.expect_int());
            return;
        }
        TORCH_INTERNAL_ASSERT(self->is_meta());
        maybe_resize_storage_meta(self, std::move(new_size_bytes));
    }

    static void maybe_resize_storage_meta(TensorImpl* self, c10::SymInt new_size_bytes) {
        // It does not make sense to try to resize a storage
        // to hold 0 elements, and this can break
        // if storage_offset is positive but
        // new_size is 0, so just bail in that case
        // (same comment is in Resize.h)
        if (self->sym_numel() == 0) {
            return;
        }

        const Storage& storage = self->unsafe_storage();
        if (!storage) {
            TORCH_INTERNAL_ASSERT(0, "NYI, this should only be Caffe2");
        } else if (new_size_bytes > storage.sym_nbytes()) {
            resize_bytes_meta(storage.unsafeGetStorageImpl(), std::move(new_size_bytes));
        }
    }

    inline void maybe_resize_storage_cpu(TensorImpl* self, size_t new_size_bytes) {
        // It does not make sense to try to resize a storage
        // to hold 0 elements, and this can break
        // if storage_offset is positive but
        // new_size is 0, so just bail in that case
        // (same comment is in cuda/Resize.h)
        if (self->numel() == 0) {
            return;
        }

        const Storage& storage = self->unsafe_storage();
        if (!storage) {
            auto new_storage = c10::make_intrusive<StorageImpl>(
                StorageImpl::use_byte_size_t(),
                new_size_bytes,
                c10::GetCPUAllocator(),
                true);
            self->set_storage_keep_dtype(std::move(new_storage));
        } else if (new_size_bytes > storage.nbytes()) {
            resize_bytes_cpu(storage.unsafeGetStorageImpl(), new_size_bytes);
        }
    }

    void resize_bytes_cpu(StorageImpl* storage, size_t size_bytes) {
        TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");

        at::DataPtr new_data;
        if (size_bytes != 0) {
            new_data = storage->allocator()->allocate(size_bytes);
        }
        const at::DataPtr& old_data = storage->data_ptr();
        const auto old_capacity = storage->nbytes();
        const auto copy_capacity = std::min(size_bytes, old_capacity);
        if (old_data != nullptr && copy_capacity > 0) {
            memcpy(new_data.get(), old_data.get(), copy_capacity);
        }
        storage->set_data_ptr_noswap(std::move(new_data));
        storage->set_nbytes(size_bytes);
    }
    ```

    - numpy
    ```python
    @array_function_dispatch(_resize_dispatcher)
    def resize(a, new_shape):
        """
        Return a new array with the specified shape.

        If the new array is larger than the original array, then the new
        array is filled with repeated copies of `a`.  Note that this behavior
        is different from a.resize(new_shape) which fills with zeros instead
        of repeated copies of `a`.
        """
        if isinstance(new_shape, (int, nt.integer)):
            new_shape = (new_shape,)

        a = ravel(a)

        new_size = 1
        for dim_length in new_shape:
            new_size *= dim_length
            if dim_length < 0:
                raise ValueError(
                    'all elements of `new_shape` must be non-negative'
                )

        if a.size == 0 or new_size == 0:
            # First case must zero fill. The second would have repeats == 0.
            return np.zeros_like(a, shape=new_shape)

        repeats = -(-new_size // a.size)  # ceil division
        a = concatenate((a,) * repeats)[:new_size]

        return reshape(a, new_shape)
    ```


# 四、对比分析

Pytorch 中 set_ 方法的 source 除了可以传入 Tensor 还可以是 Storage，Paddle 没有对应的公开数据结构，所以 source 只能支持 Tensor。从 Pytorch 的实现可以看出 set_ 方法与 as_strided 较为类似。

Pytorch 中 resize_ 方法在 deterministic 模式下，新的元素初始化为固定值：float 或 complex 初始化为 NaN，int 初始化为最大值；而 Numpy 中，新元素初始化为 0。为兼容 Numpy 可以考虑增加参数 fill_zero，默认为 False，为 True 时新元素初始化为 0。


# 五、设计思路与实现方案

## 命名与参数设计
API `paddle.Tensor.set_(source=None, shape=None, stride=None, offset=0)`
paddle.Tensor.set_
----------------------
参数
:::::::::
- source (Tensor|optional) - 用来参照设置的 Tensor。默认值为 None，表示设置为形状是 [0] 的 Empty Tensor，此时 `stride` 也被设为 [0]。
- shape (list|tuple|optional) - 指定的新的 shape。默认值为 None，此时若 `source` 不为 None，取值为 `source` 的 shape。
- stride (list|tuple|optional) - 指定的新的 stride。默认值为 None，此时若 `source` 不为 None 且 `shape` 为 None，取值为 `source` 的 stride；若 `source` 不为 None 且 `shape` 不为 None 时取值为 paddle.empty(shape) 的 stride。
- offset (int) - 指定的新的 offset。默认值为 0。
:::::::::
- Tensor 返回参照 source 修改了 shape，stride，offset 后的 `x`。

API `paddle.Tensor.resize_(shape, fill_zero=False)`
paddle.Tensor.resize_
----------------------
参数
:::::::::
- shape (list|tuple) - 指定的新形状。
- fill_zero (bool|optional) - 当需要增加新元素时，指定是否将新元素初始化为 0。默认值为 False。
:::::::::
- Tensor 返回修改了 shape 后的 `x`。


## 底层OP设计
paddle.Tensor.set_

    ```cpp
    template <typename Context>
    void SetKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& source,
                   const std::vector<int64_t>& dims,
                   const std::vector<int64_t>& stride,
                   int64_t offset,
                   DenseTensor* out) {
        AsStridedKernel<Context>(dev_ctx, source, dims, stride, offset, out);
    }
    ```

在 ops.yaml 中配置 op : set 的 inplace : (x -> out)

## API实现方案
1. paddle.Tensor.set_

    ```python
    def set_(x, source=None, shape=None, stride=None, offset=0):
        if source is None:
            source = paddle.empty([0], dtype=x.dtype)
            shape = [0]
            stride = [0]
        if stride is None:
            if shape is None:
                stride = source.strides
            else:
                stride = paddle.empty(shape).strides
        if shape is None:
            shape = source.shape
            
        return _C_ops.set_(x, source, shape, stride, offset)
    ```

2. paddle.Tensor.resize_

    ```python
    def resize_(x, shape, fill_zero=False):
        new_size = math.prod(shape)
        old_size = x.numel()
        if (new_size > old_size) and fill_zero:
            repeats = -(-new_size // old_size)
            tmp = paddle.concat((x,) + (x,) * (repeats-1) * paddle.zeros_like(x))[:new_size]
            return x.set_(tmp, shape)

    return x.set_(x, shape)
    ```

# 六、测试和验收的考量

测试case：

paddle.Tensor.set_
- 正确性验证：可以与 as_strided 的结果对齐；
  - 不同参数组合；
  - 验证不同dtype类型：`bfloat16`，`float16`，`float32`，`float64`，`bool`，`int8`，`int16`，`int32`，`int64`，`uint8`，`complex64`，`complex128`；
  - 验证不同计算设备：覆盖 CPU 和 GPU 等实现；
单测位于test/legacy_test/test_set_inplace_api.py

paddle.Tensor.resize_
- 正确性验证：可以与 NumPy 的结果对齐；
  - old_size <= new_size ；
  - old_size > new_size ；
  - 验证不同dtype类型：`bfloat16`，`float16`，`float32`，`float64`，`bool`，`int8`，`int16`，`int32`，`int64`，`uint8`，`complex64`，`complex128`；
  - 验证不同计算设备：覆盖 CPU 和 GPU 等实现；
单测位于test/legacy_test/test_resize_inplace_api.py

# 七、可行性分析和排期规划

2024/09 完成 API 设计文档和主体实现；
2024/10 完成单测；

# 八、影响面
丰富 paddle API，对其他模块没有影响