# paddle.incubate.sparse.reshape 设计文档

| API名称                                                    | paddle.incubate.sparse.reshape                | 
|----------------------------------------------------------|-----------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | PeachML                                       | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-09-05                                    | 
| 版本号                                                      | V1.0                                          | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                       | 
| 文件名                                                      | 20220905_api_design_for_sparse_reshape.md<br> | 

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，针对 Paddle 的两种稀疏 Tensor 格式 COO 与 CSR ，都需新增 reshape 的计算逻辑，
一共需要新增 2个 kernel 的前向与反向，其中 CSR 的 kernel 需支持 2D/3D Tensor，COO 的 kernel 需支持任意维度的 Tensor。

## 3、意义

支持稀疏tensor的reshape操作，丰富基础功能，提升稀疏tensor的API完整度。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中无相关实现

```c
Tensor reshape(const Tensor& self, IntArrayRef proposed_shape) {
  if (self.is_sparse()) {
    AT_ERROR("reshape is not implemented for sparse tensors");
  }
```

## scipy

scipy中转换为coo再进行reshape

```python
def reshape(self, *args, **kwargs):
    """reshape(self, shape, order='C', copy=False)
    Gives a new shape to a sparse matrix without changing its data.
    Parameters
    ----------
    shape : length-2 tuple of ints
        The new shape should be compatible with the original shape.
    order : {'C', 'F'}, optional
        Read the elements using this index order. 'C' means to read and
        write the elements using C-like index order; e.g., read entire first
        row, then second row, etc. 'F' means to read and write the elements
        using Fortran-like index order; e.g., read entire first column, then
        second column, etc.
    copy : bool, optional
        Indicates whether or not attributes of self should be copied
        whenever possible. The degree to which attributes are copied varies
        depending on the type of sparse matrix being used.
    Returns
    -------
    reshaped_matrix : sparse matrix
        A sparse matrix with the given `shape`, not necessarily of the same
        format as the current object.
    See Also
    --------
    numpy.matrix.reshape : NumPy's implementation of 'reshape' for
                           matrices
    """
    # If the shape already matches, don't bother doing an actual reshape
    # Otherwise, the default is to convert to COO and use its reshape
    shape = check_shape(args, self.shape)
    order, copy = check_reshape_kwargs(kwargs)
    if shape == self.shape:
        if copy:
            return self.copy()
        else:
            return self

    return self.tocoo(copy=copy).reshape(shape, order=order, copy=False)

```

coo reshape实现如下

```python
def reshape(self, *args, **kwargs):
    shape = check_shape(args, self.shape)
    order, copy = check_reshape_kwargs(kwargs)

    # Return early if reshape is not required
    if shape == self.shape:
        if copy:
            return self.copy()
        else:
            return self

    nrows, ncols = self.shape

    if order == 'C':
        # Upcast to avoid overflows: the coo_matrix constructor
        # below will downcast the results to a smaller dtype, if
        # possible.
        dtype = get_index_dtype(maxval=(ncols * max(0, nrows - 1) + max(0, ncols - 1)))

        flat_indices = np.multiply(ncols, self.row, dtype=dtype) + self.col
        new_row, new_col = divmod(flat_indices, shape[1])
    elif order == 'F':
        dtype = get_index_dtype(maxval=(nrows * max(0, ncols - 1) + max(0, nrows - 1)))

        flat_indices = np.multiply(nrows, self.col, dtype=dtype) + self.row
        new_col, new_row = divmod(flat_indices, shape[0])
    else:
        raise ValueError("'order' must be 'C' or 'F'")

    # Handle copy here rather than passing on to the constructor so that no
    # copy will be made of new_row and new_col regardless
    if copy:
        new_data = self.data.copy()
    else:
        new_data = self.data

    return self.__class__((new_data, (new_row, new_col)),
                          shape=shape, copy=False)
```


## paddle DenseTensor

1. -1 表示这个维度的值是从x的元素总数和剩余维度推断出来的。因此，有且只有一个维度可以被设置为-1。
2. 0 表示实际的维数是从x的对应维数中复制出来的，因此shape中0的索引值不能超过x的rank。
3. 给定一个形状为[2,4,6]的三维张量x，目标形状为[6,8]，则将x变换为形状为[6,8]的2-D张量，且x的数据保持不变。
4. 给定一个形状为[2,4,6]的三维张量x，目标形状为[2,3,-1,2]，则将x变换为形状为[2,3,4,2]
   的4-D张量，且x的数据保持不变。在这种情况下，目标形状的一个维度被设置为-1，这个维度的值是从x的元素总数和剩余维度推断出来的。
5. 给定一个形状为[2,4,6]的三维张量x，目标形状为[-1,0,3,2]，则将x变换为形状为[2,4,3,2]
   的4-D张量，且x的数据保持不变。在这种情况下，0对应位置的维度值将从x的对应维数中复制,-1对应位置的维度值由x的元素总数和剩余维度推断出来。

代码如下

```c
DenseTensor& DenseTensor::Resize(const DDim& dims) {
  meta_.dims = dims;
  return *this;
}
```

但是此处是DenseTensor的。

# 四、对比分析

为了适配paddle phi库的设计模式，需自行设计实现方式

# 五、方案设计

## 命名与参数设计

在 paddle/phi/kernels/sparse/cpu/reshape_kernel.cc 和 paddle/phi/kernels/sparse/gpu/reshape_kernel.cu 中， kernel设计为

```    
template <typename T, typename Context>
void ReshapeCooKernel(const Context& dev_ctx,
                      const SparseCooTensor& x,
                      const phi::IntArray& shape,
                      SparseCooTensor* out)
```

```
template <typename T, typename Context>
void ReshapeCsrKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const phi::IntArray& shape,
                      SparseCsrTensor* out) 
```


在 paddle/phi/kernels/sparse/cpu/reshape_grad_kernel.cc 和 paddle/phi/kernels/sparse/gpu/reshape_grad_kernel.cu 中， 反向kernel设计为

```    
template <typename T, typename Context>
void ReshapeCooGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const SparseCooTensor& dout,
                          SparseCooTensor* dx)
```

```
template <typename T, typename Context>
void ReshapeCsrGradKernel(const Context& dev_ctx,
                          const SparseCsrTensor& x,
                          const SparseCsrTensor& dout,
                          SparseCsrTensor* dx)
```

在paddle/phi/api/yaml/sparse_ops.yaml中新增对应API， ReshapeInferMeta 是针对DenseTensor的，这里可以复用。

```yaml
- op : reshape
  args : (Tensor x,  IntArray shape)
  output : Tensor(out)
  infer_meta :
    func : ReshapeInferMeta
  kernel :
    func : reshape_coo{sparse_coo -> sparse_coo},
           reshape_csr{sparse_csr -> sparse_csr}
    layout : x
  backward : reshape_grad
```

在 paddle/phi/api/yaml/sparse_backward.yaml 中新增对应API

```yaml
- backward_op : reshape_grad
  forward : reshape(Tensor x, IntArray shape) -> Tensor(out)
  args : (Tensor x, Tensor out_grad)
  output : Tensor(x_grad)
  infer_meta :
    func : UnchangedInferMeta
    param : [x]
  kernel :
    func : reshape_coo_grad {sparse_coo, sparse_coo -> sparse_coo},
           reshape_csr_grad {sparse_csr, sparse_csr -> sparse_csr}
```

## 底层OP设计

反向kernel可以使用前向kernel来实现。对于Csr格式，转换成Coo格式再进行处理。
coo 前向kernel中，想像那些非零值存储在DenseTensor中, 先展平成一维DenseTensor, 计算出非零值的location位置，
再通过目标形状的DenseTensor的stride计算出非零值在新Tensor中的index。


## API实现方案

对于SparseCsrTensor，将csr格式转换成coo格式再进行运算，然后转换回csr格式输出。

对于SparseCooTensor，直接进行运算。目前只支持针对 `sparse_dim` 部分的维度进行reshape。

# 六、测试和验收的考量

测试考虑的case如下：

- 不同shape、维度、0/-1的正确性
- 不同 `sparse_dim`

# 七、可行性分析及规划排期

方案主要自行实现核心算法。可行。

# 八、影响面

为独立新增op，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
