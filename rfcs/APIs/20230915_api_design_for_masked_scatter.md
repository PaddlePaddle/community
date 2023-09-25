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
import numpy

def masked_scatter(x, mask, value, inplace=False):
    """
    利用现有api实现masked_scatter功能
    """
    if paddle.in_dynamic_mode():
        if mask.shape != x.shape:
            mask = paddle.broadcast_to(mask, shape=x.shape)
        # make sure the true nums in mask is <= the nums of value
        assert mask.sum() <= value.numel(), f'mask true nums must be <= value size, but got mask true nums is {mask.sum().item()}, value size is {value.numel().item()}'
        # make sure the dtype of x and source is the same
        assert x.dtype == value.dtype, f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'

        indexs = tuple(item.squeeze() for item in paddle.where(mask))
        
        if inplace:
            return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()])
        else:
            return paddle.index_put(x, indexs, value.flatten()[:mask.sum()])
    else:
        """
        经过测试，静态图模式下(当x的shape中含有-1)广播操作失效, 所以静态图下需要避免这种情况, x的形状必须显式地指定
        """
        # make sure mask.shape == x.shape
        assert -1 not in x.shape, f"in static graph mode, we don't support broadcast the mask to x whose shape has -1, but got x.shape:{x.shape}"
        # make sure the dtype of x and source is the same
        assert x.dtype == value.dtype, f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'
        mask = paddle.broadcast_to(mask, shape=x.shape)
        
        indexs = tuple(item.squeeze() for item in paddle.where(mask))
        return paddle.index_put(x, indexs, value.flatten()[:mask.sum()])
```
## 初步测试
### 动态图测试
测试代码如下：
```python
def test_dynamic(inplace=False):
    class Net(paddle.nn.Layer):
        def __init__(self):
            super(Net, self).__init__()
            self.linear = paddle.nn.Linear(4, 4)
        
        def forward(self, x):
            y = self.linear(x)
            return masked_scatter(y, mask, b, inplace)


    a = paddle.randn([3,4])
    print("a:", a)
    mask = paddle.randn([3,4])
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
主要测试以下几种情况：
#### inplace

```
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[ 0.95377302,  0.71991599, -0.64002633, -1.18859971],
        [ 0.14633510, -0.26224178,  0.84816700,  0.68756837],
        [ 0.64852357, -0.34401020, -1.08389294, -0.54117757]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
res:  Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[ 1.        ,  0.53523803,  2.        ,  0.58861065],
        [ 3.        ,  0.24224952,  4.        , -0.35548905],
        [ 5.        , -0.36515144,  6.        ,  0.64138633]])
```
#### outplace
```
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[-0.96355069,  0.58669102, -0.23947759,  1.59143174],
        [-1.01221061,  0.08593263, -0.34094945, -0.47603396],
        [ 2.30312920,  0.44003361,  0.00515982,  0.79982501]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
res:  Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=False,
       [[ 1.        ,  0.21128452,  2.        ,  1.40798700],
        [ 3.        , -0.93409020,  4.        ,  0.95187712],
        [ 5.        ,  2.35421300,  6.        , -1.22600794]])
```
#### mask中true的个数大于value的个数
```
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[ 0.75833082,  0.04611617, -1.38131642,  1.13807058],
        [-0.49769101, -0.12536772,  0.79886371, -0.92195636],
        [ 0.93623048, -0.98690981, -0.26431829, -1.08623803]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 125, in <module>
    test_dynamic(False)
  File "D:\PythonProjects\community\test.py", line 117, in test_dynamic
    res = net(a)
          ^^^^^^
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\nn\layer\layers.py", line 1348, in __call__
    return self.forward(*inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\PythonProjects\community\test.py", line 105, in forward
    return masked_scatter(y, mask, b, inplace)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\PythonProjects\community\test.py", line 43, in masked_scatter
    assert mask.sum() <= value.numel(), f'mask true nums must be <= value size, but got mask true nums is {mask.sum().item()}, value size is {value.numel().item()}'
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: mask true nums must be <= value size, but got mask true nums is 6, value size is 5
```
#### x和value的dtype不一致
```
a: Tensor(shape=[3, 4], dtype=float32, place=Place(cpu), stop_gradient=True,
       [[-2.07125783,  0.27555400,  0.29631290,  0.89497519],
        [-0.54692996, -0.73247778, -0.26578471,  0.55248821],
        [ 0.79015601,  0.98371685, -1.17396295, -0.04731252]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(cpu), stop_gradient=True,
       [True , False, True , False])
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 125, in <module>
    test_dynamic(False)
  File "D:\PythonProjects\community\test.py", line 117, in test_dynamic
    res = net(a)
          ^^^^^^
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\nn\layer\layers.py", line 1348, in __call__
    return self.forward(*inputs, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\PythonProjects\community\test.py", line 105, in forward
    return masked_scatter(y, mask, b, inplace)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\PythonProjects\community\test.py", line 45, in masked_scatter
    assert x.dtype == value.dtype, f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: x and value must have the same dtype, but got x dtype is paddle.float32, value dtype is paddle.int32
```
### 静态图测试
#### 测试代码
```python
def test_static():
    paddle.enable_static()
    exe = paddle.static.Executor()
    train_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(train_program, startup_program):
        mask_ = paddle.static.data(name='mask', shape=[None, 5], dtype='bool')
        value_ = paddle.static.data(name='value', shape=[None, 5], dtype='float32')
        data = paddle.static.data(name='X', shape=[3, 4], dtype='float32')
        hidden = paddle.static.nn.fc(data, 5)
        out = masked_scatter(hidden, mask_, value_)
        loss = paddle.mean(out)
        paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)

    exe.run(startup_program)

    x = numpy.random.random(size=(3, 4)).astype('float32')
    mask = numpy.random.random(size=(1, 5))
    mask = mask>0.5
    print("x: ", x)
    print("mask: ",mask)
    v = numpy.ones((3, 5)).astype('float32')
    loss_data, out= exe.run(train_program, feed={"X": x,"mask": mask, "value":v}, fetch_list=[loss.name, out.name])
    print("res: ", out)
    # compiled_prog = paddle.static.CompiledProgram(train_program)
    # loss_data, out= exe.run(compiled_prog, feed={"X": x,"mask": mask, "value":v}, fetch_list=[loss.name, out.name])
    # print("result of masked_scatter(compiled): ", out)

```
主要测试以下几种情况：
#### 正常情况（静态图只有outplace）
```
I0925 19:36:53.425036 10144 program_interpreter.cc:140] New Executor is Running.
x:  [[0.6100007  0.04530565 0.12533963 0.00868342]
 [0.97731996 0.72944784 0.04382805 0.9545004 ]
 [0.66368145 0.6143798  0.17013946 0.6249167 ]]
mask:  [[False False  True  True  True]]
I0925 19:36:53.485648 10144 interpreter_util.cc:605] Standalone Executor is Used.
res:  [[0.47807267 0.39527717 1.         1.         1.        ]
 [1.4488099  0.22996539 1.         1.         1.        ]
 [0.96345973 0.20303118 1.         1.         1.        ]]
```
#### x的shape中含有-1
```
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 126, in <module>
    test_static()
  File "D:\PythonProjects\community\test.py", line 79, in test_static
    out = masked_scatter(hidden, mask_, value_)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\PythonProjects\community\test.py", line 58, in masked_scatter
    assert -1 not in x.shape, f"in static graph mode, we don't support broadcast the mask to x whose shape has -1, but got x.shape:{x.shape}"
           ^^^^^^^^^^^^^^^^^
AssertionError: in static graph mode, we don't support broadcast the mask to x whose shape has -1, but got x.shape:(-1, 5)
```
#### x和value的dtype不一致
```
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 125, in <module>
    test_static()
  File "D:\PythonProjects\community\test.py", line 78, in test_static
    out = masked_scatter(hidden, mask_, value_)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\PythonProjects\community\test.py", line 60, in masked_scatter
    assert x.dtype == value.dtype, f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: x and value must have the same dtype, but got x dtype is paddle.float32, value dtype is paddle.int32
```
#### mask中true的个数大于value的个数
```
I0925 19:43:28.299003  3496 program_interpreter.cc:140] New Executor is Running.
x:  [[0.93393874 0.54505134 0.92284834 0.01059747]
 [0.26887837 0.70674247 0.24665648 0.01458587]
 [0.8900972  0.10662988 0.931161   0.7416481 ]]
mask:  [[ True  True  True  True  True]]
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 125, in <module>
    test_static()
  File "D:\PythonProjects\community\test.py", line 90, in test_static
    loss_data, out= exe.run(train_program, feed={"X": x,"mask": mask, "value":v}, fetch_list=[loss.name, out.name])
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\base\executor.py", line 1635, in run
    res = self._run_impl(
          ^^^^^^^^^^^^^^^
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\base\executor.py", line 1842, in _run_impl
    ret = new_exe.run(list(feed.keys()), return_numpy)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\MyAPP\miniconda\Lib\site-packages\paddle\base\executor.py", line 799, in run
    tensors = self._new_exe.run(feed_names)._move_to_list()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: In user code:

    File "D:\PythonProjects\community\test.py", line 125, in <module>
      test_static()
    File "D:\PythonProjects\community\test.py", line 78, in test_static
      out = masked_scatter(hidden, mask_, value_)
    File "D:\PythonProjects\community\test.py", line 64, in masked_scatter
      return paddle.index_put(x, indexs, value.flatten()[:mask.sum()])
    File "E:\MyAPP\miniconda\Lib\site-packages\paddle\tensor\manipulation.py", line 4939, in index_put
      helper.append_op(
    File "E:\MyAPP\miniconda\Lib\site-packages\paddle\base\layer_helper.py", line 45, in append_op
      return self.main_program.current_block().append_op(*args, **kwargs)
    File "E:\MyAPP\miniconda\Lib\site-packages\paddle\base\framework.py", line 4368, in append_op
      op = Operator(
    File "E:\MyAPP\miniconda\Lib\site-packages\paddle\base\framework.py", line 2906, in __init__
      for frame in traceback.extract_stack():

    InvalidArgumentError: The value (10) of the non-singleton dimension does not match the corresponding value (15) in shape for expand_v2 op.
      [Hint: Expected vec_in_dims[i] == expand_shape[i], but received vec_in_dims[i]:10 != expand_shape[i]:15.] (at ..\paddle/phi/kernels/impl/expand_kernel_impl.h:65)
      [operator < index_put > error]
```
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

- `x (Tensor)`: 输入的张量，支持的数据类型为float16、float32、float64、int32、int64、bool，需要根据mask进行赋值操作，静态图模式下，x的形状暂时不支持在运行过程中推导，请传入形状固定的x。
- `mask (Tensor, bool)`: 用于指定填充位置的布尔值掩码张量，与 input 张量形状相同，或者可以广播成input张量的形状。
- `value (Tensor)`: 待填充的张量，支持的数据类型为float16、float32、float64、int32、int64、bool，其中元素的数量应该不少于mask中True的个数，且元素数据类型要跟x中元素数据类型保持一致。
- `name (str，可选)` :一般无需设置，默认值为 None。
> 注：x所支持参数类型参考了`paddle.index_put`这个api，与其保持一致
## 底层OP设计

依赖已有OP(broadcast_to / flatten)实现，无需实现新的底层OP。

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