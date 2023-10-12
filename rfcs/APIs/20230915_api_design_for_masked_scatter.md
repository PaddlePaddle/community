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

- `paddle.broadcast_to`：将输入广播到指定形状

- `paddle.where`：查找mask中值为true对应位置的索引。
- `paddle.index_put`：根据索引和值对待填充tensor对应位置进行赋值。

`masked_scatter`函数需要支持的一个重要特性是数据自动广播，此处参考了`paddle.where`中对于自动广播的实现。

### paddle.where中广播实现思路

`paddle.where`的源码如下所示：

```python
def where(condition, x=None, y=None, name=None):
    if np.isscalar(x):
        x = paddle.full([1], x, np.array([x]).dtype.name)

    if np.isscalar(y):
        y = paddle.full([1], y, np.array([y]).dtype.name)

    if x is None and y is None:
        return nonzero(condition, as_tuple=True)

    if x is None or y is None:
        raise ValueError("either both or neither of x and y should be given")

    condition_shape = list(condition.shape)
    x_shape = list(x.shape)
    y_shape = list(y.shape)

    if x_shape == y_shape and condition_shape == x_shape:
        broadcast_condition = condition
        broadcast_x = x
        broadcast_y = y
    else:
        zeros_like_x = paddle.zeros_like(x)
        zeros_like_y = paddle.zeros_like(y)
        zeros_like_condition = paddle.zeros_like(condition)
        zeros_like_condition = paddle.cast(zeros_like_condition, x.dtype)
        cast_cond = paddle.cast(condition, x.dtype)

        broadcast_zeros = paddle.add(zeros_like_x, zeros_like_y)
        broadcast_zeros = paddle.add(broadcast_zeros, zeros_like_condition)
        broadcast_x = paddle.add(x, broadcast_zeros)
        broadcast_y = paddle.add(y, broadcast_zeros)
        broadcast_condition = paddle.add(cast_cond, broadcast_zeros)
        broadcast_condition = paddle.cast(broadcast_condition, 'bool')

    if in_dynamic_or_pir_mode():
        return _C_ops.where(broadcast_condition, broadcast_x, broadcast_y)
    else:
        check_variable_and_dtype(condition, 'condition', ['bool'], 'where')
        check_variable_and_dtype(
            x,
            'x',
            ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'where',
        )
        check_variable_and_dtype(
            y,
            'y',
            ['uint16', 'float16', 'float32', 'float64', 'int32', 'int64'],
            'where',
        )
        helper = LayerHelper("where", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type='where',
            inputs={
                'Condition': broadcast_condition,
                'X': broadcast_x,
                'Y': broadcast_y,
            },
            outputs={'Out': [out]},
        )

        return out
```

可以看出，当输入的x和y形状不一致时，先通过`zeros_like`函数分别构造一个跟x、y形状一样的变量`zeros_like_x`、`zeros_like_y`，然后利用`add`函数将他们相加，在相加的过程中利用加法的底层实现已经自动实现了广播，最后得到的`broadcast_zeros`的形状就是最后需要广播到的形状，再拿它和x、y直接相加，就可以在相加过程中自动实现广播。

### masked_scatter的初步实现

初步实现如下：

```python
import paddle
import numpy

def masked_scatter(x, mask, value, inplace=False):
    """
    利用现有api实现masked_scatter功能
    """
    # make sure the dtype of x and source is the same
    assert x.dtype == value.dtype, f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'

    if paddle.in_dynamic_mode():
        if mask.shape != x.shape:
            mask = paddle.broadcast_to(mask, shape=x.shape)
        # make sure the true nums in mask is <= the nums of value
        assert mask.sum() <= value.numel(), f'mask true nums must be <= value size, but got mask true nums is {mask.sum().item()}, value size is {value.numel().item()}'

        indexs = tuple(item.squeeze() for item in paddle.where(mask))

        if inplace:
            return paddle.index_put_(x, indexs, value.flatten()[:mask.sum()])
        else:
            return paddle.index_put(x, indexs, value.flatten()[:mask.sum()])
    else:
        zeros_like_x = paddle.zeros_like(x)
        mask = paddle.add(paddle.cast(mask, x.dtype), zeros_like_x)
        mask = paddle.cast(mask, "bool")
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
    b = paddle.ones([2,4], dtype="float32")
    
    net = Net()
    res = net(a)

    loss = paddle.mean(paddle.pow(res-paddle.ones_like(res), 2))
    loss.backward()
    print("res: ",res)
```
主要测试以下几种情况：
#### inplace

```
W1012 16:38:27.197413 12648 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.7
W1012 16:38:27.529726 12648 gpu_resources.cc:149] device: 0, cuDNN Version: 8.4.
a: Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[ 0.42409050,  0.77523839, -2.36436272, -1.23578155],
        [ 0.44526708, -0.51622814,  0.00783824,  0.44433489],
        [-1.55585825,  0.72679478,  0.75432545,  0.72628176]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(gpu:0), stop_gradient=True,
       [True , False, True , False])
res:  Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [[ 1.        , -2.59757781,  1.        , -2.37750435],
        [ 1.        , -0.11681330,  1.        ,  0.56991023],
        [ 1.        ,  2.51356053,  1.        ,  0.67361248]])
```
#### outplace
```
W1012 16:39:25.472004 12380 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.7
W1012 16:39:25.473963 12380 gpu_resources.cc:149] device: 0, cuDNN Version: 8.4.
a: Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[ 0.36675033,  0.45838809,  0.52961969, -0.81227469],
        [-0.40587279,  0.08904818,  1.51585436,  0.85850716],
        [ 0.51168704,  0.68378007, -0.01682868, -1.81072354]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(gpu:0), stop_gradient=True,
       [True , False, True , False])
res:  Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=False,
       [[ 1.        ,  0.88243854,  1.        , -0.13420233],
        [ 1.        , -0.28148487,  1.        ,  1.31290603],
        [ 1.        ,  1.36912191,  1.        , -1.06872141]])
```
#### mask中true的个数大于value的个数
```
W1012 16:40:12.676610 17524 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.7
W1012 16:40:12.680604 17524 gpu_resources.cc:149] device: 0, cuDNN Version: 8.4.
a: Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[-1.28732932, -0.50087887,  0.75080359,  0.46962994],
        [ 0.28776711, -0.14213948,  0.39556077,  1.99971640],
        [ 0.45063889,  0.72602481,  0.14427330,  0.40330032]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(gpu:0), stop_gradient=True,
       [True , False, True , False])
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 153, in <module>
    test_dynamic(False)
  File "D:\PythonProjects\community\test.py", line 145, in test_dynamic
    res = net(a)
  File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\nn\layer\layers.py", line 1343, in __call__
    return self.forward(*inputs, **kwargs)
  File "D:\PythonProjects\community\test.py", line 133, in forward
    return masked_scatter(y, mask, b, inplace)
  File "D:\PythonProjects\community\test.py", line 88, in masked_scatter
    assert mask.sum() <= value.numel(), f'mask true nums must be <= value size, but got mask true nums is {mask.sum().item()}, value size is {value.numel().item()}'
AssertionError: mask true nums must be <= value size, but got mask true nums is 6, value size is 4
```
#### x和value的dtype不一致
```
W1012 16:40:47.663931 18240 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.7
W1012 16:40:47.665930 18240 gpu_resources.cc:149] device: 0, cuDNN Version: 8.4.
a: Tensor(shape=[3, 4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
       [[-0.15918727, -0.95514601,  0.46210659, -1.34718204],
        [ 1.31591105,  0.55861431,  0.74736464,  0.02223789],
        [-2.12608886,  1.51154649, -0.04368414,  1.97998977]])
mask:  Tensor(shape=[4], dtype=bool, place=Place(gpu:0), stop_gradient=True,
       [True , False, True , False])
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 153, in <module>
    test_dynamic(False)
  File "D:\PythonProjects\community\test.py", line 145, in test_dynamic
    res = net(a)
  File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\nn\layer\layers.py", line 1343, in __call__
    return self.forward(*inputs, **kwargs)
  File "D:\PythonProjects\community\test.py", line 133, in forward
    return masked_scatter(y, mask, b, inplace)
  File "D:\PythonProjects\community\test.py", line 74, in masked_scatter
    assert x.dtype == value.dtype, f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'
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
        data = paddle.static.data(name='X', shape=[None, 4], dtype='float32')
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
I1012 16:51:37.847046 15656 program_interpreter.cc:185] New Executor is Running.
W1012 16:51:37.847046 15656 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.7
W1012 16:51:37.850046 15656 gpu_resources.cc:149] device: 0, cuDNN Version: 8.4.
x:  [[0.3413213  0.36220944 0.9810611  0.9069664 ]
 [0.4400902  0.34666905 0.94219476 0.7365747 ]
 [0.18846968 0.56770104 0.08312963 0.6576356 ]]
mask:  [[False False False  True  True]]
I1012 16:51:40.134096 15656 interpreter_util.cc:608] Standalone Executor is Used.
res:  [[-1.1278851   0.37371185 -0.94038844  1.          1.        ]
 [-1.0448693   0.36276507 -0.8637674   1.          1.        ]
 [-0.52354705 -0.11333936 -0.98203325  1.          1.        ]]
```
#### x的shape中含有-1
```
I1012 16:52:50.811165  4360 program_interpreter.cc:185] New Executor is Running.
W1012 16:52:50.811165  4360 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.7
W1012 16:52:50.813164  4360 gpu_resources.cc:149] device: 0, cuDNN Version: 8.4.
x:  [[0.4929188  0.9807315  0.36060813 0.3813048 ]
 [0.23349337 0.63630897 0.6916962  0.38356784]
 [0.7763553  0.49821827 0.8376672  0.56760925]]
mask:  [[ True False False False  True]]
I1012 16:52:53.049795  4360 interpreter_util.cc:608] Standalone Executor is Used.
res:  [[ 1.          0.41184133 -0.7128209  -0.7933337   1.        ]
 [ 1.          0.04515322 -0.14544843 -0.6089552   1.        ]
 [ 1.         -0.00228108 -0.21314141 -0.8098481   1.        ]]
```
#### x和value的dtype不一致
```
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 153, in <module>
    test_static()
  File "D:\PythonProjects\community\test.py", line 106, in test_static
    out = masked_scatter(hidden, mask_, value_)
  File "D:\PythonProjects\community\test.py", line 74, in masked_scatter
    assert x.dtype == value.dtype, f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'
AssertionError: x and value must have the same dtype, but got x dtype is paddle.float32, value dtype is paddle.int32
```
#### mask中true的个数大于value的个数
```
I1012 16:55:54.661437 10100 program_interpreter.cc:185] New Executor is Running.
W1012 16:55:54.662438 10100 gpu_resources.cc:119] Please NOTE: device: 0, GPU Compute Capability: 8.6, Driver API Version: 12.2, Runtime API Version: 11.7
W1012 16:55:54.664407 10100 gpu_resources.cc:149] device: 0, cuDNN Version: 8.4.
x:  [[0.84189963 0.24295884 0.33678734 0.6711854 ]
 [0.9159404  0.03022701 0.44019344 0.14213431]
 [0.8592184  0.4729605  0.11157154 0.5277734 ]]
mask:  [[ True  True  True  True  True]]
Traceback (most recent call last):
  File "D:\PythonProjects\community\test.py", line 153, in <module>
    test_static()
  File "D:\PythonProjects\community\test.py", line 118, in test_static
    loss_data, out= exe.run(train_program, feed={"X": x,"mask": mask, "value":v}, fetch_list=[loss.name, out.name])
  File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\base\executor.py", line 1620, in run
    res = self._run_impl(
  File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\base\executor.py", line 1827, in _run_impl
    ret = new_exe.run(list(feed.keys()), return_numpy)
  File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\base\executor.py", line 788, in run
    tensors = self._new_exe.run(feed_names)._move_to_list()
ValueError: In user code:

    File "D:\PythonProjects\community\test.py", line 153, in <module>
      test_static()
    File "D:\PythonProjects\community\test.py", line 106, in test_static
      out = masked_scatter(hidden, mask_, value_)
    File "D:\PythonProjects\community\test.py", line 93, in masked_scatter
      return paddle.index_put(x, indexs, value.flatten()[:mask.sum()])
    File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\tensor\manipulation.py", line 4936, in index_put
      helper.append_op(
    File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\base\layer_helper.py", line 44, in append_op
      return self.main_program.current_block().append_op(*args, **kwargs)
    File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\base\framework.py", line 4404, in append_op
      op = Operator(
    File "E:\MyAPP\miniconda\envs\paddledev\lib\site-packages\paddle\base\framework.py", line 2971, in __init__
      for frame in traceback.extract_stack():

    InvalidArgumentError: The value (10) of the non-singleton dimension does not match the corresponding value (15) in shape for expand kernel.
      [Hint: Expected out_shape[i] == expand_shape[i], but received out_shape[i]:10 != expand_shape[i]:15.] (at C:\home\workspace\Paddle\paddle\phi\kernels\gpu\expand_kernel.cu:57)
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
        经过测试，静态图模式下(当x的shape中含有-1)broasdcast_to广播操作失效。但是可以借助乘法来间接实现广播效果
        """
        # make sure the dtype of x and source is the same
        assert x.dtype == value.dtype, f'x and value must have the same dtype, but got x dtype is {x.dtype}, value dtype is {value.dtype}'
        mask_ = ((x * x) + 1.) * mask > 0
        
        indexs = tuple(item.squeeze() for item in paddle.where(mask_))
        return paddle.index_put(x, indexs, value.flatten()[:mask_.sum()])
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