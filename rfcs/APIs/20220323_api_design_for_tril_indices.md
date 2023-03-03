# paddle.tril_indices设计文档


|API名称 | paddle.tril_indices |
|---|---|
|提交作者 | 哆啦A梦 |
|提交时间 | 2022-03-23 |
|版本号 | V1.0 |
|依赖飞桨版本 | develop |
|文件名 | 20220323_api_design_for_tril_indices.md |


# 一、概述
## 1、相关背景
`tril_indices`与`triu_indice`对应，`tril_indices(rows, cols, offset)`返回2行x列`tensor`,分别表示行数为`rows`列数为`cols`的二维矩阵下三角元素的行列索引。  
如果offset = 0，表示主对角线; 如果offset是正数，表示主对角线之上的对角线; 如果offset是负数，表示主对角线之下的对角线。offset的范围为[-rows+1,cols-1] 

## 2、功能目标

在飞桨中增加`paddle.tril_indices`这个API。

## 3、意义

飞桨将直接提供`tril_indices`这个API,高效运行在CPU和GPU后端

# 二、飞桨现状
飞桨提供tril(triu)函数，输入二维矩阵，返回下三角元素为1，其余部分元素为0的矩阵  
接口为`paddle.tril(input, diagonal=0, name=None)`[源码](https://github.com/PaddlePaddle/Paddle/blob/release/2.2/python/paddle/tensor/creation.py#L582)  
使用示例:
```python
data = np.arange(1, 13, dtype="int64").reshape(3,-1)
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

x = paddle.to_tensor(data)

tril1 = paddle.tensor.tril(x)
# array([[ 1,  0,  0,  0],
#        [ 5,  6,  0,  0],
#        [ 9, 10, 11,  0]])

# example 2, positive diagonal value
tril2 = paddle.tensor.tril(x, diagonal=2)
# array([[ 1,  2,  3,  0],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

```
飞桨目前没有直接提供`tril_indices`API，也无法通过组合API的方式得到。


# 三、业内方案调研
PyTorch和Numpy中都有tril_indices这个API
## PyTorch

### 实现解读
pytorch 中python接口配置为：  

```python
func: tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  dispatch:
    CPU: tril_indices_cpu
    CUDA: tril_indices_cuda
```

在PyTorch中，tril_indices是由C++和cuda实现的，CPU核心代码为：  

```c++
Tensor tril_indices_cpu(
    int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  if (!dtype_opt.has_value()) {
    dtype_opt = ScalarType::Long;
  }

  check_args(row, col, layout_opt);

  auto tril_size = get_tril_size(row, col, offset);

  // create an empty Tensor with correct size
  auto result = at::native::empty_cpu({2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  // The following three approaches result in very little performance
  // differences. Hence, the 2nd option is taken for simpler code, and to return
  // contiguous tensors. Refer to #14904 for more details.
  //
  // 1. sequential RAM access: fill row coordinates first, then columns. This
  //    results in two for-loop and more arithmetic operations.
  //
  // 2. interleaved RAM access: fill in index coordinates one by one, which
  //    jumps between the two output Tensor rows in every iteration.
  //
  // 3. sequential RAM + transpose: create an n X 2 Tensor, fill the Tensor
  //    sequentially, and then transpose it.
  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, result.scalar_type(), "tril_indices", [&]() -> void {
    // fill the Tensor with correct values
    scalar_t* result_data = result.data_ptr<scalar_t>();
    int64_t i = 0;

    scalar_t r = std::max<int64_t>(0, -offset), c = 0;
    while (i < tril_size) {
      result_data[i] = r;
      result_data[tril_size + i++] = c;

      // move to the next column and check if (r, c) is still in bound
      c += 1;
      if (c > r + offset || c >= col) {
        r += 1;
        c = 0;
        // NOTE: not necessary to check if r is less than row here, because i
        // and tril_size provide the guarantee
      }
    }
  });
```
上面代码的主要工作是：

1. 检查输入（row,col,非负,layout_opt是否有值）；
2. get_tril_size()计算返回的tensor列方向维度：
3. 开辟返回值空间，按规律给二维tensor赋值（同时注意行列坐标是否越界）

在GPU运行时rows*cols的规模要小于2^59,避免在计算过程中内存溢出
GPU核心代码为：
```c++
void tril_indices_kernel(scalar_t * tensor,
                         int64_t row_offset,
                         int64_t m_first_row,
                         int64_t col,
                         int64_t trapezoid_size,
                         int64_t tril_size) {
  int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_index < tril_size) {
    int64_t r, c;
    if (linear_index < trapezoid_size) {
      // the coordinate is within the top trapezoid
      get_coordinate_in_tril_trapezoid(m_first_row, linear_index, r, c);
    } else {
      // the coordinate falls in the bottom rectangle
      auto surplus = linear_index - trapezoid_size;
      // add the height of trapezoid: m_last_row (col) - m_first_row + 1
      r = surplus / col + col - m_first_row + 1;
      c = surplus % col;
    }
    r += row_offset;

    tensor[linear_index] = r;
    tensor[linear_index + tril_size] = c;
  }
}

// Some Large test cases for the fallback binary search path is disabled by
// default to speed up CI tests and to avoid OOM error. When modifying the
// implementation, please enable them in test/test_cuda.py and make sure they
// pass on your local server.
Tensor tril_indices_cuda(
    int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  check_args(row, col, layout_opt);

  auto tril_size = get_tril_size(row, col, offset);
  auto tensor = empty_cuda({2, tril_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  if (tril_size > 0) {
    auto m_first_row = offset > 0 ?
      std::min<int64_t>(col, 1 + offset) : // upper bounded by col
      row + offset > 0; // either 0 or 1
    auto trapezoid_row_offset = std::max<int64_t>(0, -offset);
    auto rectangle_row_offset = trapezoid_row_offset + col - m_first_row + 1;
    int64_t rectangle_size = 0;
    if (rectangle_row_offset < row) {
      rectangle_size = (row - rectangle_row_offset) * col;
    }

    dim3 dim_block = cuda::getApplyBlock();
    dim3 dim_grid;
    // using tril_size instead of tensor.numel(), as each thread takes care of
    // two elements in the tensor.
    TORCH_CHECK(
      cuda::getApplyGrid(tril_size, dim_grid, tensor.get_device()),
      "unable to get dim grid");

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, tensor.scalar_type(), "tril_indices_cuda", [&] {
      tril_indices_kernel<<<
          dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        tensor.data_ptr<scalar_t>(),
        trapezoid_row_offset,
        m_first_row,
        col,
        tril_size - rectangle_size,
        tril_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }

  return tensor;
}
```
上述代码流程与CPU端代码相同，只因为GPU的并行模式特殊，需实现不同block的核函数，以tril_indices_kernel()实现，  
外层函数需要对每个block负责的绝对坐标进行计算
### 使用示例

```python
>>> import torch
>>> a = torch.tril_indices(3, 3)
>>> a
tensor([[0, 1, 1, 2, 2, 2],
        [0, 0, 1, 0, 1, 2]])

>>> a = torch.tril_indices(4, 3, -1)
>>> a
tensor([[1, 2, 2, 3, 3, 3],
        [0, 0, 1, 0, 1, 2]])

>>> a = torch.tril_indices(4, 3, 1)
>>> a
tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        [0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
```

## NumPy

### 实现解读

调用接口为`numpy.tril_indices(n,k=0,m=None)`，n为矩阵行数，m为矩阵列数（可选），k为偏移，正数向右上方向偏移，  
返回二维数组为指定元素的行列

```python
def tril_indices(n, k, m):
    tri_ = tri(n, m, k=k, dtype=bool)

    return tuple(broadcast_to(inds, tri_.shape)[tri_]
                 for inds in indices(tri_.shape, sparse=True))
```
上述代码调用函数`tri()`获得一个n*m维矩阵，其下三角元素为True，其余元素为False.  
通过indices()函数取出此矩阵的行列下标,用broadcast_to()函数展开坐标.原理如下:

```python
>>> c = np.indices((3,3), sparse=True)
(array([[0],
       [1],
       [2]]), array([[0, 1, 2]]))
>>> for i in c
...     print(i)
[[0]
 [1]
 [2]]
[[0 1 2]]
>>> for i in c:
...     j = np.broadcast_to(i,tri_.shape)[tri_]
...     print(j)
...
[0 1 1 2 2 2]
[0 0 1 0 1 2]
```

### 使用示例

```python
>>> import numpy
>>> il1 = np.tril_indices(4)
>>> il1
(array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]), array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3]))
>>> il2 = np.tril_indices(4，2)
>>> il2
(array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]), array([0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]))
>>> a = np.arange(16).reshape(4, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
>>> a[il1]
array([ 0, 4, 5, 8, 9, 10, 12, 13, 14, 15])
>>> a[il1] = -1
>>> a
array([[-1,  1,  2,  3],
       [-1, -1,  6,  7],
       [-1, -1, -1, 11],
       [-1, -1, -1, -1]])
>>> a[il2] = 10
>>> a
array([[-10, -10, -10,   3],
       [-10, -10, -10, -10],
       [-10, -10, -10, -10],
       [-10, -10, -10, -10]])

```
使用tril_indices接口可以取出指定的对角线元素行列坐标，从而修改矩阵中指定的对角线元素值

# 四、对比分析
 `numpy.tril_indices`比`torch.tril_indices`功能相同,但实现方式不同  
 pytorch中使用数学方式直接计算需要输出的下标，而numpy中使用一系列的函数巧妙地进行输出   
 分析numpy的实现巧妙，但是中间变量占用空间大，在规模大时会影响性能

# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.tril_indices(row, col, offset,dtype=None)`，产生一个2行x列的二维数组存放指定下三角区域的，第一行为行坐标，第二行为列坐标

参数类型要求：

- `row`、`col`、`offset`的类型是`int`
- 输出`Tensor`的dtype默认参数为None时使用'int64'，否则以用户输入为准

## 底层OP设计

在`paddle/fluid/operators/tril_indices_op.cc`添加tril_indices算子的描述，

在`paddle/phi/infermeta/nultiary.h`中声明形状推断的函数原型，在`paddle/phi/infermeta/nultiary.cc`中实现。

```c++
void TrilIndicesInferMeta(const int& row,
                       const int& col,
                       const int& offset,
                       MetaTensor* out);
```

在`paddle/phi/kernels/tril_indices_kernel.h`中声明核函数的原型  

```c++
template <typename Context>
void TrilIndicesKernel( const Context& dev_ctx,
                        const int& row,
                        const int& col,
                        const int& offset,
                        DataType dtype,
                        DenseTensor* out);
```

分别在 `paddle/phi/kernels/cpu/tril_indices_kernel.cc``paddle/phi/kernels/gpu/tril_indices_kernel.cu`注册和实现核函数  
实现逻辑借鉴pytorch直接计算下标。  
CPU实现逻辑：计算输出数组大小，开辟空间，遍历每个位置赋值行列坐标。  
GPU实现逻辑：计算输出数组大小，计算每个block负责的原始行列，按照输出数组大小进行平均的任务划分，实现每个block的赋值kernel  

## python API实现方案

在`python/paddle/fluid/layers/tensor.py`中增加`tril_indices`函数,添加英文描述

```python
def tril_indices(row, col, offset, dtype=None):
    # ...
    # 参数检查,非整数类型转换成整数类型，给出提示
    # ...
    if dtype == None :
        dtype == int
    # ...
    # 调用核函数
    TrilIndicesKernel(dev_ctx,row,col,offset,dtype,out)
    # ...
    return out
```
## 单测及文档填写
在` python/paddle/fluid/tests/unittests/`中添加`test_tril_indices.py`文件进行单测,测试代码使用numpy计算结果后对比，与numpy对齐    
在` docs/api/paddle/`中添加中文API文档

# 六、测试和验收的考量

- 输入合法性及有效性检验；

- 对比与Numpy的结果的一致性：
  不同情况（m>n || n>m || offset\in {1-rows , cols-1} || offset\notin {1-rows , cols-1})

- CPU、GPU测试。

# 七、可行性分析和排期规划
T 确定指导文档，熟悉paddle算子编程风格  
T + 1 week 完成 CPU 端代码  
T + 2 weeks 完成 GPU 端代码   
T + 3 weeks 完成 单元测试并提交  

# 八、影响面
tril_indices是独立API，不会对其他API产生影响。

# 名词解释
无

# 附件及参考资料
无
