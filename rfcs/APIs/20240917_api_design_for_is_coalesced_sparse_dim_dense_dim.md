# paddle.Tensor.is_coalesced，paddle.Tensor.sparse_dim，paddle.Tensor.dense_dim 设计文档

|API名称 | paddle.Tensor.is_coalesced /paddle.Tensor.sparse_dim /paddle.Tensor.dense_dim | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | NKNaN | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2024-09-17 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20240917_api_design_for_is_coalesced_sparse_dim_dense_dim.md<br> | 


# 一、概述
## 1、相关背景
[NO.25 为 Paddle 新增 is_coalesced / sparse_dim / dense_dim API](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_7th/【Hackathon%207th】个人挑战赛—框架开发任务合集.md#no25-为-paddle-新增-is_coalescedsparse_dimdense_dim-api)

## 2、功能目标
- 实现 paddle.Tensor.is_coalesced 做为 Tensor 的方法使用。如果 Tensor 是一个已合并的稀疏张量，则返回 True，否则返回 False。
- 实现 paddle.Tensor.sparse_dim 做为 Tensor 的方法使用。返回稀疏张量中稀疏维度的数量。
- 实现 paddle.Tensor.dense_dim 做为 Tensor 的方法使用。返回稀疏张量中密集维度的数量。


## 3、意义
新增 paddle.Tensor.is_coalesced /paddle.Tensor.sparse_dim /paddle.Tensor.dense_dim 方法，丰富 paddle sparse API

# 二、飞桨现状
对于 paddle.Tensor.is_coalesced 目前在 SparseCooTensor 的底层实现中有对应的 coalesced_ 标识变量；
对于 paddle.Tensor.sparse_dim 和 paddle.Tensor.dense_dim 目前在 SparseCooTensor 的底层实现中有对应的方法，而在 SparseCsrTensor 中没有；

# 三、业内方案调研

### PyTorch
- PyTorch 中的 torch.Tensor.is_coalesced [API文档](https://pytorch.org/docs/stable/generated/torch.Tensor.is_coalesced.html)
- PyTorch 中的 torch.Tensor.sparse_dim [API文档](https://pytorch.org/docs/stable/generated/torch.Tensor.sparse_dim.html)
- PyTorch 中的 torch.Tensor.dense_dim [API文档](https://pytorch.org/docs/stable/generated/torch.Tensor.dense_dim.html)

### Scipy
Scipy.sparse 模块没有相关方法的实现

### 实现方法
- is_coalesced
    
    ```cpp
    bool is_coalesced_sparse(const SparseTensor& self) {
        return get_sparse_impl(self)->coalesced();
    }

    bool coalesced() const {
        return coalesced_;
    }
    ```

- sparse_dim

    ```cpp
    int64_t sparse_dim_sparse(const SparseTensor& self) {
        return get_sparse_impl(self)->sparse_dim();
    }

    // SparseTensorImpl.h
    int64_t sparse_dim() const {
        return sparse_dim_;
    }

    // SparseCsrTensorImpl.h
    inline int64_t sparse_dim() const noexcept {
        return 2;
    }
    ```


- dense_dim

    ```cpp
    int64_t dense_dim_sparse(const SparseTensor& self) {
        return get_sparse_impl(self)->dense_dim();
    }

    // SparseTensorImpl.h
    int64_t dense_dim() const {
        return dense_dim_;
    }

    // SparseCsrTensorImpl.h
    inline int64_t dense_dim() const noexcept {
        return values_.dim() - batch_dim() - block_dim() - 1;
    }

    inline int64_t batch_dim() const noexcept {
        return crow_indices_.dim() - 1;
    }

    inline int64_t block_dim() const noexcept {
        return (layout_ == kSparseBsr || layout_ == kSparseBsc ? 2 : 0);
    }
    ```



# 四、对比分析

pytorch 的 SparseCooTensor 与 paddle 的底层实现基本一致，因此 is_coalesced 可以直接参考 pytorch，最后在 pybind 中定义调用 SparseTensor 的 coalesced() Tensor方法。

对于 sparse_dim 和 dense_dim，paddle 的 SparseCooTensor 类有对应接口但 SparseCsrTensor 类尚未实现，需要对应增加接口后，再在 pybind 中调用。


# 五、设计思路与实现方案

## 命名与参数设计
API `paddle.Tensor.is_coalesced(x)`
paddle.Tensor.is_coalesced
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
:::::::::
- bool 返回稀疏张量 `x` 是否为合并后的 SparseCooTensor 。

API `paddle.Tensor.sparse_dim(x)`
paddle.Tensor.sparse_dim
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
:::::::::
- int 返回稀疏张量 `x` 的稀疏维度的数量。

API `paddle.Tensor.dense_dim(x)`
paddle.Tensor.dense_dim
----------------------
参数
:::::::::
- x (Tensor) - 输入 Tensor。
:::::::::
- int 返回稀疏张量 `x` 的密集维度的数量。


## 底层设计

- paddle.Tensor.is_coalesced - 不涉及

- paddle.Tensor.sparse_dim - 涉及 SparseCsrTensor 增加对应接口

    ```cpp
    int32_t sparse_dim() const;

    int32_t SparseCsrTensor::sparse_dim() const {
        return 2;
    }
    ```

- paddle.Tensor.dense_dim - 涉及 SparseCsrTensor 增加对应接口

    ```cpp
    int32_t dense_dim() const;

    int32_t SparseCsrTensor::dense_dim() const {
        int32_t nze_dim = non_zero_elements_.dims().size();
        int32_t batch_dim = non_zero_crows_.dims().size() - 1;
        // layout of SparseCsrTensor has not been implemented yet
        // int32_t block_dim =  = (layout_ == kSparseBsr || layout_ == kSparseBsc ? 2 : 0);
        int32_t block_dim = 0;
        return nze_dim - batch_dim - block_dim - 1;
    }
    ```

## API实现方案 - pybind
1. paddle.Tensor.is_coalesced
    
    ```cpp
    static PyObject* tensor_method_is_coalesced(TensorObject* self,
                                                PyObject* args,
                                                PyObject* kwargs) {
        EAGER_TRY
        if (self->tensor.is_sparse_coo_tensor()) {
            auto sparse_coo_tensor =
                std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
            return ToPyObject(sparse_coo_tensor->coalesced());
        } else {
            return ToPyObject(false);
        }
        EAGER_CATCH_AND_THROW_RETURN_NULL
    }
    ```

2. paddle.Tensor.sparse_dim

    ```cpp
    static PyObject* tensor_method_sparse_dim(TensorObject* self,
                                              PyObject* args,
                                              PyObject* kwargs) {
        EAGER_TRY
        if (self->tensor.is_sparse_coo_tensor()) {
            auto sparse_coo_tensor =
                std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
            return ToPyObject(sparse_coo_tensor->sparse_dim());
        } else if (self->tensor.is_sparse_csr_tensor()) {
            auto sparse_csr_tensor =
                std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
            return ToPyObject(sparse_csr_tensor->sparse_dim());
        } else {
            return ToPyObject(0);
        }
        EAGER_CATCH_AND_THROW_RETURN_NULL
    }
    ```
3. paddle.Tensor.dense_dim

    ```cpp
    static PyObject* tensor_method_dense_dim(TensorObject* self,
                                              PyObject* args,
                                              PyObject* kwargs) {
        EAGER_TRY
        if (self->tensor.is_sparse_coo_tensor()) {
            auto sparse_coo_tensor =
                std::dynamic_pointer_cast<phi::SparseCooTensor>(self->tensor.impl());
            return ToPyObject(sparse_coo_tensor->dense_dim());
        } else if (self->tensor.is_sparse_csr_tensor()) {
            auto sparse_csr_tensor =
                std::dynamic_pointer_cast<phi::SparseCsrTensor>(self->tensor.impl());
            return ToPyObject(sparse_csr_tensor->dense_dim());
        } else {
            return ToPyObject(self->tensor.shape().size());
        }
        EAGER_CATCH_AND_THROW_RETURN_NULL
    }
    ```


# 六、测试和验收的考量

测试case：

paddle.Tensor.is_coalesced，paddle.Tensor.sparse_dim，paddle.Tensor.dense_dim
- 正确性验证：用 Scipy 实现对应方法，结果对齐；
  - 不同 shape；
  - 不同 dtype 类型：验证 'bool' ，'float16'，'float32'， 'float64' ，'int8'，'int16'，'int32'，'int64'，'uint8'，'complex64'，'complex128'；
  - 不同设备；


# 七、可行性分析和排期规划

2024/09 完成 API 主体设计与实现；
2024/10 完成单测；

# 八、影响面
丰富 paddle sparse API，对其他模块没有影响
