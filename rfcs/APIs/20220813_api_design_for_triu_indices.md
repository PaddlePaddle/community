# paddle.triu_indices设计文档


|API名称 | paddle.triu_indices |
|---|---|
|提交作者 | Rayman的团队 |
|提交时间 | 2022-08-13 |
|版本号 | V1.0 |
|依赖飞桨版本 | develop |
|文件名 | 20220813_api_design_for_triu_indices.md |


# 一、概述
## 1、相关背景
`triu_indices` 能获取一个2维矩阵的上三角元素的索引，其输出 Tensor 的 shape 为$[2, N]$，相当于有两行，第一行为 上三角元素的行索引，第二行为下三角元素的列索引。调用方式与`tril_indices(rows, cols, offset)`对应。offset的范围为$[-rows+1,cols-1]$。

## 2、功能目标

在Paddle框架中增加`paddle.triu_indices`这个API。

## 3、意义

Paddle将提供高效的`triu_indices`API供用户直接调用。

# 二、飞桨现状
飞桨目前没有提供`triu_indices`这个API，且无法通过API组合的方式间接实现其功能。

相关接口：
1. 飞桨目前提供了triu函数，输入矩阵和对角线的参数，返回矩阵上三角部分，其余部分元素为零。调用接口为`paddle.triu(input, diagonal=0, name=None)`
[源码](https://github.com/PaddlePaddle/Paddle/blob/release/2.3/python/paddle/tensor/creation.py#L674)
[文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/triu_cn.html#triu)

2. 飞桨提供了tril_indices函数，与期望实现的triu_indices类似，但其返回的是下三角元素索引[源码](https://github.com/PaddlePaddle/Paddle/blob/6d31dc937704380efe2dee97716c3da47b7060f1/python/paddle/tensor/creation.py#L1721)。

```python
import paddle
            
# example 1, default offset value
data1 = paddle.tril_indices(4,4,0)
print(data1)
# [[0, 1, 1, 2, 2, 2, 3, 3, 3, 3], 
#  [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]]
# example 2, positive offset value
data2 = paddle.tril_indices(4,4,2)
print(data2)
# [[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], 
#  [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]]
# example 3, negative offset value
data3 = paddle.tril_indices(4,4,-1)
print(data3)
# [[ 1, 2, 2, 3, 3, 3],
#  [ 0, 0, 1, 0, 1, 2]]
```

# 三、业内方案调研
PyTorch和Numpy中都有triu_indices这个API
## PyTorch

### 实现解读
pytorch 中接口配置为: [在线文档](https://pytorch.org/docs/stable/generated/torch.triu_indices.html?highlight=triu_indices#torch.triu_indices)

```python
  torch.triu_indices(row, col, offset=0, *, dtype=torch.long, device='cpu', layout=torch.strided) → Tensor
```

在PyTorch中，triu_indices是由C++和CUDA实现的，其中CPU核心代码为：  

```c++
Tensor triu_indices_cpu(
    int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  if (!dtype_opt.has_value()) {
    dtype_opt = ScalarType::Long;
  }

  check_args(row, col, layout_opt);

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);

  // create an empty Tensor with correct size
  auto result = at::native::empty_cpu({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  AT_DISPATCH_ALL_TYPES_AND(kBFloat16, result.scalar_type(), "triu_indices", [&]() -> void {
    // fill the Tensor with correct values
    scalar_t* result_data = result.data_ptr<scalar_t>();
    int64_t i = 0;
    // not typing std::max with scalar_t as it could be an unsigned type
    // NOTE: no need to check if the returned value of std::max overflows
    // scalar_t, as i and triu_size act as a guard.
    scalar_t c = std::max<int64_t>(0, offset), r = 0;
    while (i < triu_size) {
      result_data[i] = r;
      result_data[triu_size + i++] = c;

      // move to the next column and check if (r, c) is still in bound
      c += 1;
      if (c >= col) {
        r += 1;
        // not typing std::max with scalar_t as it could be an unsigned type
        // NOTE: not necessary to check if c is less than col or overflows here,
        // because i and triu_size act as a guard.
        c = std::max<int64_t>(0, r + offset);
      }
    }
  });

  return result;
}
```
CPU端的代码主要逻辑是：

1. 对入参进行检查
2. 复用get_tril_size()函数，通过总维度减去下三角区域得到需要的tensor维度
3. 创建空的Tensor并赋值得到正确的输出。

GPU核心代码为：
```c++
template <typename scalar_t>
__global__
void triu_indices_kernel(scalar_t * tensor,
                         int64_t col_offset,
                         int64_t m_first_row,
                         int64_t col,
                         int64_t rectangle_size,
                         int64_t triu_size) {
  int64_t linear_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_index < triu_size) {
    int64_t r, c;
    if (linear_index < rectangle_size) {
      // the coordinate is within the top rectangle
      r = linear_index / col;
      c = linear_index % col;
    } else {
      // the coordinate falls in the bottom trapezoid
      get_coordinate_in_triu_trapezoid(
        m_first_row, linear_index - rectangle_size, r, c);
      r += rectangle_size / col;
    }

    c += col_offset;
    tensor[linear_index] = r;
    tensor[linear_index + triu_size] = c;
  }
}

// Some Large test cases for the fallback binary search path is disabled by
// default to speed up CI tests and to avoid OOM error. When modifying the
// implementation, please enable them in test/test_cuda.py and make sure they
// pass on your local server.
Tensor triu_indices_cuda(
    int64_t row, int64_t col, int64_t offset, c10::optional<ScalarType> dtype_opt,
    c10::optional<Layout> layout_opt, c10::optional<Device> device_opt, c10::optional<bool> pin_memory_opt) {
  check_args(row, col, layout_opt);

  auto triu_size = row * col - get_tril_size(row, col, offset - 1);
  auto tensor = empty_cuda({2, triu_size}, dtype_opt, layout_opt, device_opt, pin_memory_opt);

  if (triu_size > 0) {
    // # of triu elements in the first row
    auto m_first_row = offset > 0 ?
      std::max<int64_t>(col - offset, 0) : // upper bounded by col
      col;

    // size of the top rectangle
    int64_t rectangle_size = 0;
    if (offset < 0) {
      rectangle_size = std::min<int64_t>(row, -offset) * col;
    }

    dim3 dim_block = cuda::getApplyBlock();
    dim3 dim_grid;

    // using triu_size instead of tensor.numel(), as each thread takes care of
    // two elements in the tensor.
    TORCH_CHECK(
      cuda::getApplyGrid(triu_size, dim_grid, tensor.get_device()),
      "unable to get dim grid");

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, tensor.scalar_type(), "triu_indices_cuda", [&] {
      triu_indices_kernel<<<
          dim_grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        tensor.data_ptr<scalar_t>(),
        std::max<int64_t>(0, offset),
        m_first_row,
        col,
        rectangle_size,
        triu_size);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }

  return tensor;
}
```

上述CUDA代码是计算逻辑的GPU端实现，整体上未进行特殊的优化，计算逻辑清晰简洁。


### 使用示例

```python
>>> import torch
>>> a = torch.triu_indices(3, 3)
>>> a
tensor([[0, 0, 0, 1, 1, 2],
        [0, 1, 2, 1, 2, 2]])

>>> a = torch.triu_indices(4, 3, -1)
>>> a
tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3],
        [0, 1, 2, 0, 1, 2, 1, 2, 2]])

>>> a = torch.triu_indices(4, 3, 1)
>>> a
tensor([[0, 0, 1],
        [1, 2, 2]])
```

## NumPy

### 实现解读

调用接口为`numpy.triu_indices(n,k=0,m=None)`，n为矩阵行数，m为矩阵列数（可选），k为偏移，正数向右上方向偏移，  
返回二维数组为指定元素的行列

```python
def triu_indices(n, k=0, m=None):
    tri_ = ~tri(n, m, k=k - 1, dtype=bool)

    return tuple(broadcast_to(inds, tri_.shape)[tri_]
                 for inds in indices(tri_.shape, sparse=True))
```
上述代码调用函数`tri()`获得一个n*m维矩阵，其下上角元素为True，其余元素为False. 其底层通过umath库实现。
后通过indices()函数取出此矩阵的行列下标,用broadcast_to()函数展开坐标.

### 使用示例

```python
>>> import numpy as np
>>> iu1 = np.triu_indices(4)
>>> iu2 = np.triu_indices(4, 2)

Here is how they can be used with a sample array:

>>> a = np.arange(16).reshape(4, 4)
>>> a
array([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])

Both for indexing:

>>> a[iu1]
array([ 0,  1,  2, ..., 10, 11, 15])

And for assigning values:

>>> a[iu1] = -1
>>> a
array([[-1, -1, -1, -1],
        [ 4, -1, -1, -1],
        [ 8,  9, -1, -1],
        [12, 13, 14, -1]])

These cover only a small part of the whole array (two diagonals right
of the main one):

>>> a[iu2] = -10
>>> a
array([[ -1,  -1, -10, -10],
        [  4,  -1,  -1, -10],
        [  8,   9,  -1,  -1],
        [ 12,  13,  14,  -1]])

```
使用triu_indices接口可以取出指定的对角线元素行列坐标，从而修改矩阵中指定的对角线元素值

# 四、对比分析
 `numpy.triu_indices`比`torch.triu_indices`功能相同,但实现方式略有不同  
 pytorch中根据定义直接计算需要输出的下标，而numpy中使用一系列的函数巧妙地进行输出 

 分析numpy的实现巧妙，但是中间变量占用空间大，在规模大时会影响性能，且不支持GPU加速。
 
 另外`paddle.tril_indices`是实现逻辑与pytorch类似，故`paddle.triu_indices`主体参考pytorch的实现思路保持代码一致性。

# 五、设计思路与实现方案

## 命名与参数设计
API设计为`paddle.triu_indices(row, col=None, offset=0,dtype='int64')`，产生一个2行x列的二维数组存放指定上三角区域的坐标，第一行为行坐标，第二行为列坐标

参数类型要求：

- `row`、`col`、`offset`的类型是`int`
- 输出`Tensor`的dtype默认参数为None时使用'int64'，否则以用户输入为准

## 底层OP设计

在`paddle/fluid/operators/triu_indices_op.cc`添加triu_indices算子的描述，

在`paddle/phi/infermeta/nullary.h`中声明形状推断的函数原型，在`paddle/phi/infermeta/nullary.cc`中实现。

```c++
void TriuIndicesInferMeta(const int& row,
                       const int& col,
                       const int& offset,
                       MetaTensor* out);
```

在`paddle/phi/kernels/triu_indices_kernel.h`中声明核函数的原型  

```c++
template <typename Context>
void TriuIndicesKernel( const Context& dev_ctx,
                        const int& row,
                        const int& col,
                        const int& offset,
                        DataType dtype,
                        DenseTensor* out);
```

分别在 `paddle/phi/kernels/cpu/triu_indices_kernel.cc` 和`paddle/phi/kernels/gpu/triu_indices_kernel.cu`注册和实现核函数  
实现逻辑借鉴pytorch直接计算下标。  
CPU实现逻辑：计算输出数组大小，开辟空间，遍历每个位置赋值行列坐标。  
GPU实现逻辑：计算输出数组大小，计算每个block负责的原始行列，按照输出数组大小进行平均的任务划分，实现每个block的赋值kernel。（目前pytorch版本的逻辑在device端实际存在一定的线程束分化，如有时间可以尝试进行优化。）

## python API实现方案

在`python/paddle/tensor/creation.py`中增加`triu_indices`函数，并添加英文描述

```python
def triu_indices(row, col=None, offset=0, dtype='int64'):
    # ...
    # 参数检查
    # col默认为None, 当col取None时代表输入为正方形，将row值赋给col
    # ...
    # 增加算子
    # ...
    return out
```
## 单测及文档填写
在` python/paddle/fluid/tests/unittests/`中添加`test_triu_indices.py`文件进行单测,测试代码使用numpy计算结果后对比，与numpy对齐    
在` docs/api/paddle/`中添加中文API文档

# 六、测试和验收的考量

- 输入合法性及有效性检验；

- 对比与Numpy的结果的一致性：
  不同情况 
  $（m>n || n>m || offset \in \{1-rows , cols-1\} || offset \notin \{1-rows , cols-1\})$

- CPU、GPU测试。

# 七、可行性分析和排期规划
已完成主体开发，8.21前完成单元测试并提交

# 八、影响面
triu_indices是独立API，不会对其他API产生影响。

# 名词解释
无

# 附件及参考资料
无
