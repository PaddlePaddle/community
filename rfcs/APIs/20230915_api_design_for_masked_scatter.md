# paddle.masked_scatter 设计文档

|API名称 | paddle.masked_scatter | 
|---|---|
|提交作者| ranchongzhi | 
|提交时间 | 2023-09-15 | 
|版本号 | V1.0 | 
|依赖飞桨版本| develop | 
|文件名 | 20230915_api_design_for_masked_scatter.md | 


# 一、概述
## 1、相关背景

`masked_scatter`是一个常用的api，用于根据给定的`mask`以及源`Tensor`在目标`Tensor`指定位置上进行替换操作，这个功能在序列标注等任务中经常被用到，因此，在Paddle中提供该API，方便用户使用。

## 2、功能目标

在 Paddle 框架中，新增 `paddle.masked_scatter` 对于一个`目标Tensor`，根据`mask`信息，将`源Tensor`中的值按照顺序填充到`目标Tensor`中`mask`对应为`True`的位置。

## 3、意义

该API是一个常用的API，可以方便用户使用。让用户不用自己实现该功能，提高用户的使用效率。

# 二、飞桨现状
## 组合实现
目前paddle缺少相关功能实现。只能通过 paddle 现有的 API 组合实现。主要利用的api如下：
- `paddle.broadcast_to`：将mask广播成和待填充tensor一样形状。
- `paddle.where`：查找mask中值为true对应位置的索引。
- `paddle.index_put`：根据索引和值对待填充tensor对应位置进行赋值。

具体代码实现如下：
```python
import paddle

def masked_scatter(x, mask, value, inplace=False):
    """
    利用现有api实现masked_scatter功能
    """
    # make sure the mask can be broadcastable to input
    assert paddle.broadcast_shape(mask.shape, x.shape)==x.shape, f'mask is not be broadcastable to input, mask shape is {mask.shape}, input shape is {x.shape}'
    mask = paddle.broadcast_to(mask, shape=x.shape)
    # make sure the true nums in mask is <= the nums of value
    assert mask.sum() <= value.numel(), 'mask true nums must be <= value size'
    # make sure the dtype of x and source is the same
    assert x.dtype == value.dtype, 'input and source must have the same dtype'

    indexs = tuple(item.squeeze() for item in paddle.where(mask))
    print("index of true value in mask: ", indexs)
    if inplace and paddle.in_dynamic_mode():
        return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()])
    else:
        return paddle.index_put(x, indexs, value.flatten()[:mask.sum()])
```
## 初步测试
测试的代码如下所示：
```python
class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = paddle.nn.Linear(4, 4)
    
    @paddle.jit.to_static
    def forward(self, x):
        y = self.linear(x)
        return masked_scatter(y, mask, b, inplace=True)


a = paddle.randn([3,4])
print("a:", a)
mask = paddle.to_tensor([1.,0.5,1.,0.5])
mask = mask>0.6
print("mask: ", mask)
b = paddle.to_tensor([1.,2.,3.,4.,5.,6.,7.])
 
net = Net()
res = net(a)

loss = paddle.mean(paddle.pow(res-paddle.ones_like(res), 2))
loss.backward()
print("res: ",res)
```
通过装饰器`@paddle.jit.to_static`指定动静态图模式，通过`inplace`参数指定执行inplace或outplace操作。
### 动态图测试
#### outplace
```
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[ 0.50508595, -0.57142472,  0.34164023, -1.71793330],
        [-1.04813683,  1.94749498,  0.92576098,  0.18977740],
        [ 0.24962157, -0.95671540, -0.70601028,  0.20051311]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
index of true value in mask:  (Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
       [0, 0, 1, 1, 2, 2]), Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
       [0, 2, 0, 2, 0, 2]))
res:  Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[ 1.        , -0.48612538,  2.        ,  1.98325372],
        [ 3.        , -1.27658749,  4.        , -1.09418225],
        [ 5.        ,  0.70582151,  6.        ,  0.13813046]])
```
#### inplace
```
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[-1.28495479, -0.33748436,  2.15355968, -1.56535161],
        [ 0.07834079, -0.16553184, -0.60210073,  0.51506144],
        [-0.02662235, -1.04836142, -1.66325510,  0.92996895]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
index of true value in mask:  (Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
       [0, 0, 1, 1, 2, 2]), Tensor(shape=[6], dtype=int64, place=Place(cpu), stop_gradient=True,
       [0, 2, 0, 2, 0, 2]))
res:  Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[ 1.        ,  1.03400707,  2.        , -3.30637121],
        [ 3.        , -0.07386497,  4.        ,  0.61626333],
        [ 5.        , -0.09191823,  6.        ,  0.74832195]])
```
### 静态图测试
#### outplace
```
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[-1.76391935, -0.00142353,  0.32293102,  2.12124515],
        [-1.27195692, -1.44442165,  0.40191424, -2.08972764],
        [ 0.67450720, -0.40461785, -1.49469006, -0.15822217]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
index of true value in mask:  (var squeeze_0.tmp_0 : LOD_TENSOR.shape(-1,).dtype(int64).stop_gradient(True), var squeeze_1.tmp_0 : LOD_TENSOR.shape(-1,).dtype(int64).stop_gradient(True))
I0920 15:59:18.341959  7804 program_interpreter.cc:140] New Executor is Running.
I0920 15:59:18.349983  7804 interpreter_util.cc:605] Standalone Executor is Used.
res:  Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[ 1.        ,  0.89891565,  2.        , -0.57123518],
        [ 3.        , -0.41151577,  4.        ,  0.73677796],
        [ 5.        , -0.69148386,  6.        , -0.32962719]])
```
#### inplace
静态图模式下调用`paddle.index_put_`会自动调用`paddle.index_put`，此处仍展示运行结果以及警告信息，测试代码如下：
```python
import paddle

def masked_scatter(x, mask, value, inplace=False):
    """
    利用现有api实现masked_scatter功能
    """
    # make sure the mask can be broadcastable to input
    assert paddle.broadcast_shape(mask.shape, x.shape)==x.shape, f'mask is not be broadcastable to input, mask shape is {mask.shape}, input shape is {x.shape}'
    # turn mask to bool
    mask = paddle.broadcast_to(mask, shape=x.shape)
    # make sure the true nums in mask is <= the nums of value
    assert mask.sum() <= value.numel(), 'mask true nums must be <= value size'
    # make sure the dtype of x and source is the same
    assert x.dtype == value.dtype, 'input and source must have the same dtype'

    indexs = tuple(item.squeeze() for item in paddle.where(mask))
    print("index of true value in mask: ", indexs)
    if inplace and not paddle.in_dynamic_mode():
        return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()])

class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = paddle.nn.Linear(4, 4)
    
    @paddle.jit.to_static
    def forward(self, x):
        y = self.linear(x)
        return masked_scatter(y, mask, b, inplace=True)


a = paddle.randn([3,4])
print("a:", a)
mask = paddle.to_tensor([1.,0.5,1.,0.5])
mask = mask>0.6
print("mask: ", mask)
b = paddle.to_tensor([1.,2.,3.,4.,5.,6.,7.])
 
net = Net()
res = net(a)

loss = paddle.mean(paddle.pow(res-paddle.ones_like(res), 2))
loss.backward()
print("res: ",res)

'''
这种情况会报错：
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[-0.34334219,  0.29797703,  1.57610333,  0.57713234],
        [ 1.01555479, -0.78327483,  1.50608945, -0.32091832],
        [-0.76809406, -0.49917048, -0.96261829,  0.04977092]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
index of true value in mask:  (var squeeze_0.tmp_0 : LOD_TENSOR.shape(-1,).dtype(int64).stop_gradient(True), var squeeze_1.tmp_0 : LOD_TENSOR.shape(-1,).dtype(int64).stop_gradient(True))
E:\MyAPP\miniconda\Lib\site-packages\paddle\utils\inplace_utils.py:31: UserWarning: In static graph mode, index_put_() is the same as index_put() and does not perform inplace operation.
  warnings.warn(
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 76, in <module>
    res = net(a)
          ^^^^^^
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\nn\layer\layers.py", line 1348, in __call__
    return self.forward(*inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\jit\dy2static\program_translator.py", line 480, in __call__
    return self._perform_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\jit\dy2static\program_translator.py", line 802, in _perform_call
    error_data.raise_new_exception()
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\jit\dy2static\error.py", line 452, in raise_new_exception
    raise new_exception from None
ValueError: In transformed code:

    File "D:\PythonProjects\community\test.py", line 65, in forward
        return masked_scatter(y, mask, b, inplace=True)
    File "D:\PythonProjects\community\test.py", line 49, in masked_scatter
        if inplace and not paddle.in_dynamic_mode():
    File "D:\PythonProjects\community\test.py", line 50, in masked_scatter
        print("index of true value in mask: ", indexs)
        if inplace and not paddle.in_dynamic_mode():
            return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()])
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
        # if inplace and paddle.in_dynamic_mode():
        #     return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()])

    File "E:\MyAPP\miniconda\Lib\site-packages\decorator.py", line 232, in fun
        return caller(func, *(extras + args), **kw)
    File "E:\MyAPP\miniconda\Lib\site-packages\paddle\base\wrapped_decorator.py", line 25, in __impl__
        return wrapped_func(*args, **kwargs)
    File "E:\MyAPP\miniconda\Lib\site-packages\paddle\utils\inplace_utils.py", line 41, in __impl__
        raise ValueError(

    ValueError: Sorry about what's happend. In to_static mode, index_put_'s output variable flatten_0.tmp_0_slice_0 is a viewed Tensor in dygraph. This will result in inconsistent calculation behavior between dynamic and static graphs. You mast find the location of the strided API be called, and call flatten_0.tmp_0_slice_0 = flatten_0.tmp_0_slice_0.assign().
'''
```
根据报错提示修改代码，做法是将用于填充的Tensor复制一份：
```python
# 修改前
return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()])    
# 修改后
return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()].clone())

'''
运行结果如下：
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[ 0.72673988,  1.74302280,  0.36312774, -0.89876205],
        [-0.83842933, -0.06078178,  1.21603084,  1.57265437],
        [-0.10622863,  0.21086957, -0.16041717, -0.34174833]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
index of true value in mask:  (var squeeze_0.tmp_0 : LOD_TENSOR.shape(-1,).dtype(int64).stop_gradient(True), var squeeze_1.tmp_0 : LOD_TENSOR.shape(-1,).dtype(int64).stop_gradient(True))
E:\MyAPP\miniconda\Lib\site-packages\paddle\utils\inplace_utils.py:31: UserWarning: In static graph mode, index_put_() is the same as index_put() and does not perform inplace operation.
  warnings.warn(
I0920 16:11:18.380725 20956 program_interpreter.cc:140] New Executor is Running.
I0920 16:11:18.387750 20956 interpreter_util.cc:605] Standalone Executor is Used.
res:  Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[ 1.        , -2.04496288,  2.        ,  1.41189837],
        [ 3.        ,  1.45620036,  4.        , -0.32584974],
        [ 5.        , -0.36014718,  6.        ,  0.02144548]])
'''
```
## 总结
从测试结果看来，静态图模式下，`paddle.index_put_`这个api貌似无法使用，所以只在`动态图且需要inplace操作`的情况下调用`paddle.index_put_`，其余情况都调用`paddle.index_put`。

# 三、业内方案调研

## PyTorch

PyTorch中提供了`Tensor.masked_scatter(mask, tensor)`以及`Tensor.masked_scatter_(mask, source)`两个API，分别表示`out-place`和`in-place`两种计算形式，在pytorch中的介绍如下：

```
Copies elements from source into self tensor at positions where the mask is True. Elements from source are copied into self starting at position 0 of source and continuing in order one-by-one for each occurrence of mask being True. The shape of mask must be broadcastable with the shape of the underlying tensor. The source should have at least as many elements as the number of ones in mask.
```
其中输入参数的描述如下：

- mask (BoolTensor) – the boolean mask
- source (Tensor) – the tensor to copy from
### 实现方法

在实现方法上, Pytorch 设计了两种实现方式，一种是CPU实现，一种是GPU实现，核心代码如下：

```cpp
//GPU实现
//代码路径：pytorch/aten/src/ATen/native/cuda/IndexKernel.cu
void launch_masked_scatter_kernel(
    const TensorBase &self, const TensorBase &mask,
    const TensorBase &maskPrefixSum, const TensorBase &source) {
  const auto srcSize = source.numel();
  const auto mask_cont = mask.contiguous();
  const auto mask_numel = mask.numel();

  // Use a prefix sum to determine the output locations of the masked elements
  auto maskPrefixSum_data = maskPrefixSum.mutable_data_ptr<int64_t>();
  auto mask_data = mask_cont.const_data_ptr<bool>();

  at::cuda::cub::mask_exclusive_sum(
      mask_data, maskPrefixSum_data, mask_numel);

  // Asynchronously check that the number of `1` elements present in the mask
  // must be <= the number of elements available in `src`.
  masked_scatter_size_check<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      &maskPrefixSum_data[mask_numel - 1], &mask_data[mask_numel - 1], srcSize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  auto source_contig = source.contiguous();

  auto iter = TensorIteratorConfig()
      .set_check_mem_overlap(false)
      .check_all_same_dtype(false)
      .resize_outputs(false)
      .add_output(self)
      .add_input(self)
      .add_input(mask_cont)
      .add_input(maskPrefixSum)
      .build();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      self.scalar_type(),
      "masked_scatter_",
      [&]() {
        auto source_ptr = source_contig.const_data_ptr<scalar_t>();
        gpu_kernel(
            iter, [=] GPU_LAMBDA(const scalar_t a, const bool mask, const int64_t maskPrefixSum) -> scalar_t {
              if (mask) {
                return source_ptr[maskPrefixSum];
              }
              return a;
            });
        AT_CUDA_CHECK(cudaGetLastError());
      });
}
```
对于GPU实现，PyTorch的实现思路如下：
1. 首先，获取源张量的元素数量（srcSize）以及掩码张量的连续版本（mask_cont）和元素数量（mask_numel）。
   
2. 使用前缀和（prefix sum）计算掩码元素的输出位置。maskPrefixSum是一个输入张量，用于存储掩码元素的前缀和结果。maskPrefixSum_data和mask_data分别获取了maskPrefixSum和mask_cont的数据指针。
   
3. 异步地检查掩码中为1的元素数量是否小于等于源张量中的元素数量。这是通过调用masked_scatter_size_check内核函数实现的，该函数使用CUDA流来执行。这个检查确保源张量中有足够的元素供复制到掩码为True的位置。
   
4. 将源张量转换为连续版本（source_contig），以便可以根据maskPrefixSum的偏移量从source中获取元素。
   
5. 创建一个TensorIterator对象（iter），配置它的参数，包括指定输出张量（self）和输入张量（self、mask_cont、maskPrefixSum），并构建该迭代器。
   
6. 使用AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3宏对数据类型进行分派，包括除Bool、BFloat16和Half之外的所有类型。这个宏用于在CUDA内核函数中处理不同的标量类型。
7. 在内核函数中，获取源张量的数据指针（source_ptr），然后使用gpu_kernel函数执行操作。该操作是一个Lambda函数，根据掩码值（mask）和对应位置的前缀和（maskPrefixSum）选择要返回的值。如果掩码为True，则返回source_ptr[maskPrefixSum]，否则返回输入的值（a）。
8. 最后，使用cudaGetLastError检查CUDA内核函数是否执行成功


```cpp
//CPU实现
//代码路径：pytorch/aten/src/ATen/native/cpu/IndexKernel.cpp
template <typename scalar_t>
void cpu_masked_scatter_kernel(TensorIterator& iter, const TensorBase& source) {
  std::ptrdiff_t source_cntr = 0;
  scalar_t* source_ptr = source.data_ptr<scalar_t>();
  auto numel = source.numel();

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    char* dst = data[0];
    const int64_t dst_stride = strides[0];
    char* mask = data[1];
    const int64_t mask_stride = strides[1];
    for (const auto i : c10::irange(n)) {
      auto mask_value = *reinterpret_cast<bool*>(mask + mask_stride * i);
      if (mask_value) {
        TORCH_CHECK(source_cntr < numel, "Number of elements of source < number of ones in mask");
        *(scalar_t*)(dst + dst_stride * i) = *(source_ptr);
        source_ptr++;
        source_cntr++;
      }
    }
  };
  iter.serial_for_each(loop, {0, iter.numel()});
}

void masked_scatter_kernel(TensorIterator& iter, const TensorBase& source) {
 TORCH_CHECK(iter.input_dtype() == ScalarType::Bool, "masked_scatter_ only supports boolean masks, "
    "but got mask with dtype ", iter.input_dtype());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Bool,
      ScalarType::BFloat16,
      ScalarType::Half,
      iter.dtype(),
      "masked_scatter",
      [&] {
          cpu_masked_scatter_kernel<scalar_t>(iter, source);
      });
}
```
对于CPU实现，PyTorch的实现思路如下：
1. 实现一个名为cpu_masked_scatter_kernel的模板函数，用于处理CPU上的masked_scatter操作。它接受一个TensorIterator对象和一个源张量source作为参数。
2. 在cpu_masked_scatter_kernel函数中，获取源张量的数据指针（source_ptr）和元素数量（numel）。通过调用TensorIterator::serial_for_each启动循环，对每组数据都调用loop函数。
3. loop函数中通过直接访问指针进行填充判断，数据赋值等逻辑。
4. 使用宏AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3生成针对于不同数据的类型的特化模板
5. 调用入口对mask中元素的类型进行校验。

## TensorFlow

TensorFlow中没有`masked_scatter`API的实现

## Scipy

Scipy中没有`masked_scatter`API的实现

## Numpy

Numpy中没有`masked_scatter`API的实现

# 四、对比分析

- Pytorch 自定义Kernel的方式更加高效
- Tensorflow、Scipy、Numpy中没有`masked_scatter`API的实现

# 五、设计思路与实现方案

## 命名与参数设计
```python
paddle.masked_scatter(x, mask, value)

paddle.masked_scatter_(x, mask, value)

Tensor.masked_scatter(mask, value)

Tensor.masked_scatter_(mask, value)
```
masked_scatter和masked_scatter_分别表示out-place和in-place两种计算形式。

- `x (Tensor, float16, float32，float64，int32，int64，bool)`: 输入的张量，需要根据mask进行赋值操作。
- `mask (Tensor, bool)`: 用于指定填充位置的布尔值掩码张量，与 input 张量形状相同，或者可以广播成input张量的形状。
- `value (Tensor, float16, float32，float64，int32，int64，bool)`: 待填充的张量，其中元素的数量应该不少于mask中True的个数。
- `name (str，可选)` :一般无需设置，默认值为 None。
> 注：x所支持参数类型参考了`paddle.index_put`这个api，与其保持一致
## 底层OP设计

依赖已有OP(broadcast_to / flatten)实现，无需实现新的底层Op。

## API实现方案

在 python/paddle/tensor/manipulation.py 中增加 masked_scatter 以及 masked_scatter_ 函数。初步的实现方案如下：
```python
def masked_scatter(x, mask, value, inplace=False):
    """
    利用现有api实现masked_scatter功能
    """
    # make sure the mask can be broadcastable to input
    assert paddle.broadcast_shape(mask.shape, x.shape)==x.shape, f'mask is not be broadcastable to input, mask shape is {mask.shape}, input shape is {x.shape}'
    mask = paddle.broadcast_to(mask, shape=x.shape)
    # make sure the true nums in mask is <= the nums of value
    assert mask.sum() <= value.numel(), 'mask true nums must be <= value size'
    # make sure the dtype of x and source is the same
    assert x.dtype == value.dtype, 'input and source must have the same dtype'

    indexs = tuple(item.squeeze() for item in paddle.where(mask))
    print("index of true value in mask: ", indexs)
    if inplace and paddle.in_dynamic_mode():
        return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()])
    else:
        return paddle.index_put(x, indexs, value.flatten()[:mask.sum()])
```

# 六、测试和验收的考量

1. 添加单测文件 `paddle/test/legacy_test/test_masked_scatter_op.py`。
2. 同时在 `paddle/test/legacy_test/test_inplace.py` 中新增对应的inplace api 单测


测试需要考虑的 case 如下：

- 输入的mask和input的形状不一致，但是可以broadcast
- 检查算子计算结果的正确性，以pytorch为参考
- 测试在进行反向梯度计算时结果的正确性
- 错误检查：输入x不满足要求时,能否正确抛出错误

# 七、可行性分析和排期规划

方案主要利用paddle现有api完成，工期上可以满足在当前版本周期内开发完成。

# 八、影响面
新增 API，对其他模块无影响

# 名词解释

# 附件及参考资料
[paddle.index_put_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/index_put__cn.html#cn-api-paddle-index-put)

[TORCH.TENSOR.MASKED_SCATTER_](https://pytorch.org/docs/2.0/generated/torch.Tensor.masked_scatter_.html#torch.Tensor.masked_scatter_)