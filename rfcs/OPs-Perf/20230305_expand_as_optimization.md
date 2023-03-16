# 标题

标题如：Transpose OP性能优化设计文档
| 基本信息                                                   | 内容                                                         |
| ---------------------------------------------------------  | ------------------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">   | [Timber-Ye](https://github.com/Timber-Ye)、[BrianQian1999](https://github.com/BrianQian1999)                                               |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-05                                                   |
| 版本号                                                      | V1.0                                   |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| 基于PaddleDevelop版本开发                      |
| 文件名                                                       | 20230305_expand_as_op_optimization.md<br> |


# 1 背景与意义

目前 Paddle 内 `expand_as` 前向和反向算子的 GPU 实现采用 Eigen 组合的模式，缺少 GPU Kernel，性能相对不足，希望实现高性能的 GPU 计算 Kernel，为 Paddle 优化 `expand_as` op 在 GPU 上的计算性能。

## 1.1 飞桨现状

飞桨框架现有的expand_as前向算子的实现过程为：（1）首先确定每一维将被扩展的次数；（2）直接借助Eigen库的广播方法，调用`funcs::EigenBroadcast`：
``````c++
funcs::EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(place, y, x0, bcast_dims)
``````
而这一过程已经能够由Eigen库在GPU上实现。

后向算子的实现过程与之类似：（1）确定需要进行求和Reduction的维度；（2）借助Eigen库，实现`funcs::EigenBroadcastGrad`：
```c++
funcs::EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(place, x_grad, out_grad0, reduce_dims, reshape_dims);
```
其内部调用了Eigen库中针对张量的`reshape`以及`sum`方法：
```c++
out.device(dev) =
        in.reshape(reshape_dims).sum(reduce_dims).reshape(out.dimensions());
```
首先对被扩展后的高维张量进行reshape，以便后续在指定维度上进行求和，最后再将结果reshape到希望输出的形状，以此达到约归降维的目的。

下表列出了paddle框架的expand_as算子在[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)中各种case场景下的OP性能数据（测试环境：Tesla V100-32G, CUDA 11.2）。

| Case | Data type | src_shape    | dst_shape      | Paddle Forward (ms) |  Paddle Backward (ms) |   Total(ms)  |
| ---- | --------- | ------------ | -------------- | ----------          | ----------------------|--------------|
| 0    | float32   | [1785, 1]    | [1785, 128]    | 0.074236            | 0.172566              | 0.246802     |
| 1    | float32   | [5, 1, 1]    | [5, 128, 128]  | 0.082833            | 3.594770              | 3.677603     |
| 2    | float32   | [32, 807, 1] | [32, 807, 807] | 0.427489            | 1.107112              | 1.532601     |
| 3    | float16   | [1785, 1]    | [1785, 128]    | 0.049622            | 0.147476              | 0.197098     |
| 4    | float16   | [5, 1, 1]    | [5, 128, 128]  | 0.051206            | 3.039735              | 3.090941     |
| 5    | float16   | [32, 807, 1] | [32, 807, 807] | 0.407556            | 0.980826              | 1.388382     |

## 1.2 业内方案调研

### 1.2.1 Tensorflow

tensorflow中的`tf.tile` 可以用来在多个维度上重复input tensor，该方法与expand_as功能近似。其过程大致是由高维向低维进行递归扩展，每一次扩展实际上都是在进行一次数据拷贝 （[源码链接：🔗](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/tile.cc#L88-L97) ）。

`tf.tile`会创建一个新的张量来保存复制后的张量，由于复制操作涉及大量数据的读写IO运算，计算代价相对较高。

### 1.2.2 Pytorch

Pytorch中存在expand算子，其前向过程同Paddle框架现有方法一致，被视作一次Broadcast（[调用位置：🔗](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_op.h#L58-L67)），但Pytorch自身实现了广播过程的GPU Kernel（[源码链接：🔗](https://github.com/pytorch/pytorch/blob/master/caffe2/utils/math_gpu.cu#L2781-L2804)）：
``````c++
template <typename T, int D>
__global__ void BroadcastCUDAKernel(
    const int Y_size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<FIXED_DIVISOR, D> Y_dims,
    const T alpha,
    const T* X,
    T* Y) {
  CUDA_1D_KERNEL_LOOP(Y_index, Y_size) {
    int X_index = 0;
    int Y_index_val = Y_index;
#pragma unroll
    for (int i = D - 1; i >= 0; --i) {
      int d;
      FIXED_DIVISOR_DIV_MOD(Y_dims.data[i], Y_index_val, &Y_index_val, &d);
      X_index += d * X_strides.data[i];
    }
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    Y[Y_index] = __ldg(X + X_index) * alpha;
#else
    Y[Y_index] = X[X_index] * alpha;
#endif
  }
}
``````
该Kernel实现基于ElementWise方式，关键过程是找到`Y_index`与`X_index`之间的映射关系。

Expand算子的后向过程同样基于约归求和(ReduceSum)的方法（[调用位置：🔗](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/expand_op.h#L108-L116)），其GPU Kernel实现（[源码链接：🔗](https://github.com/pytorch/pytorch/blob/b8dfb45ac282a48764c192ac7d27b7d80eed8b2b/caffe2/utils/math/reduce.cu#L104-L135)）：
``````c++
template <typename T, class Reducer, int D>
__global__ void ReduceTensorCUDAKernel(
    const int inner_size,
    const SimpleArray<int, D> X_strides,
    const SimpleArray<int, D> Y_dims,
    const Reducer reducer,
    const T init,
    const T alpha,
    const T* X,
    T* Y) {
  __shared__ typename BlockReduce<T>::TempStorage temp_storage;
  const int x = blockIdx.x;
  T val = init;
  for (int y = threadIdx.x; y < inner_size; y += blockDim.x) {
    int X_index = 0;
    int Y_index = x * inner_size + y;
#pragma unroll
    for (int d = D - 1; d >= 0; --d) {
      X_index += Y_index % Y_dims.data[d] * X_strides.data[d];
      Y_index /= Y_dims.data[d];
    }
#if __CUDA_ARCH__ >= 350 || defined(USE_ROCM)
    val = reducer(val, __ldg(X + X_index));
#else
    val = reducer(val, X[X_index]);
#endif
  }
  val = BlockReduce<T>(temp_storage).Reduce(val, reducer);
  if (threadIdx.x == 0) {
    Y[x] = val * alpha;
  }
}
``````

该Kernel实现同样基于ElementWise方式，但由于`Y_index`到 `X_index`的映射为一对多，需要申请一块shared memory来记录`Y_index`所对应的所有`X_index`上的数据，并在最后对这块共享内存进行求和，最终赋值给`Y_index`所在位置。

另外针对特殊的情况，pytorch中还特别编写了RowwiseReduce（[源码链接：🔗](https://github.com/pytorch/pytorch/blob/b8dfb45ac282a48764c192ac7d27b7d80eed8b2b/caffe2/utils/math/reduce.cu#L25-L47)）、ColwiseReduce（[源码链接：🔗](https://github.com/pytorch/pytorch/blob/b8dfb45ac282a48764c192ac7d27b7d80eed8b2b/caffe2/utils/math/reduce.cu#L49-L72)）以及BothEndsReduce（[源码链接：🔗](https://github.com/pytorch/pytorch/blob/b8dfb45ac282a48764c192ac7d27b7d80eed8b2b/caffe2/utils/math/reduce.cu#L74-L102)）三个特殊的Kernel 实现。


## 1.3 对比分析

除了Paddle框架以外，[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)中还有针对Tensorflow的静态图测试脚本，下表给出的是Tensorflow框架下ExpandAs算子在各类case中的性能（测试环境：Tesla V100-32G, CUDA 11.2）：

| Case | Data type | src_shape    | dst_shape      | Tensorflow forward (ms) | Tensorflow backward (ms) | Total(ms)                |
| ---- | --------  | ---------    | ------------   | --------------          | -----------------------  | ------------------------ |
| 0    | float32   | [1785, 1]    | [1785, 128]    | 0.150479                | 0.159827                 | 0.310306                 |
| 1    | float32   | [5, 1, 1]    | [5, 128, 128]  | 0.104476                | 0.108868                 | 0.213345                 |
| 2    | float32   | [32, 807, 1] | [32, 807, 807] | 9.223847                | 9.212913                 | 18.436761                |
| 3    | float16   | [1785, 1]    | [1785, 128]    | 0.042221                | 0.044698                 | 0.086919                 |
| 4    | float16   | [5, 1, 1]    | [5, 128, 128]  | 0.024973                | 0.031283                 | 0.056257                 |
| 5    | float16   | [32, 807, 1] | [32, 807, 807] | 5.511609                | 5.298775                 | 10.810385                |

在case 2和case 5当中，Paddle比Tensorflow快近10倍；而在case 1和case 4中，Tensorflow的性能又有显著优势。另外，当数据类型从`float32`变为`float16`后，Tensorflow算子的性能有明显的提升，而相比之下，数据类型对Paddle算子性能的影响不大。

特别值得注意的是，在case 1中Tensorflow后向算子比Paddle快30倍以上，case 4则更是快了近100倍。针对Paddle框架后向算子在这两个case中的不佳表现，我们进行了进一步测试（测试环境：Tesla V100-32G, CUDA 11.2）：

| Case | Data type | src_shape    | dst_shape      | Paddle forward (ms) | Paddle backward (ms) | Total(ms)        |
| ---- | --------- | ------------ | -------------- | --------------------| ---------------------| -----------------|
| 6    | float32   | [16, 1, 1]   | [16, 807, 807] | 0.271686            | 254.208616           | 254.480303       |
| 7    | float32   | [32, 1, 1]   | [32, 256, 256] | 0.097565            | 18.539683            | 18.637249        |

综上可见，无论前向还是后向算子，Paddle与Tensorflow相比较均各有优劣。但是，Tensorflow中前向后向算子的性能差距不大，而在Paddle中，前向算子的性能通常要明显好于后向算子，也就是说**Paddle的后向算子有很大的优化空间**。所降维数空间越大，进行约归求和的数据量也就越大，这应该是导致Paddle后向算子性能存在如此差异的主要原因。

# 2 设计方案与性能预期

计划替代Paddle中所使用的Eigen库，借鉴Pytorch，采用ElementWise方式来优化expand as前向后向op。

## 2.1 关键模块与性能提升点

新增模块包含ExpandAs算子前向以及后向各自的GPU Kernel，着重提升后向过程的运算效率。

## 2.2 前向优化

ExpandAs算子的前向过程基本类似广播机制，即在需要进行扩展的维度上进行数据拷贝。从并行编程ElementWise的角度来看，可以令每一个线程处理输出张量上的每一个元素，通过Index的映射关系找出该元素在输入张量上的对应位置，然后读取数据完成赋值即可。

## 2.2.1 Host端计算流程

Host端主要是准备Index映射所需要的数据，主要包括输入、输出张量的维数、每一维的尺寸、每一维的步长等。在这之后进行Kernel Launch即可。

``````c++
...
auto x_shape_dim = x.dims(); // 输入张量在每一维上的尺寸
auto x_stride_dim = phi::stride(x_shape_dim); // 输入张量在每一维上的步长

for(int i=0; i<rank; i++){
    h_indexInfo[i] = target_shape[target_rank-i-1]; // target shape
    h_indexInfo[i + rank] = x_shape_dim[rank-i-1] == 1 ? 0 : x_stride_dim[rank-i-1]; // input stride
}
...

// kernel launch
ExpandAsForward<T>
      <<<grid_dim, block_dim, 0, stream>>>(in_data,
                                           out_data,
                                           rank,
                                           out_numel,
                                           d_indexInfo);
``````
## 2.2.2 Device端计算流程

Device端计算流程主要为以ElementWise方式寻找in_idx和out_idx之间的映射关系。某个线程在确定对应的索引之后，按照索引去global memory当中的输入张量中取值，然后赋值给输出张量的对应位置。

前向GPU Kernel当中的各个线程之间完全独立，无需申请共享内存。

``````c++
int in_idx, out_idx;

CUDA_KERNEL_LOOP_TYPE(idx, N, int64_t) {
    in_idx = idx;
    out_idx = f(in_idx);
    out_tensor[out_idx] = in_tensor[in_idx];
}

``````

## 2.3 后向优化

ExpandAs算子的后向过程基本类似约归求和过程，即将所要进行reduce的维度上的所有元素进行求和。从并行编程ElementWise的角度来看，可以令每一个Thread Block处理输出张量上的每一个元素，通过Index的映射关系找出该元素在输入张量上所对应的所有位置，然后利用Paddle内封装好的Warp级操作，调用接口快速进行Block内的全部数据的求和，最后完成赋值即可。

## 2.2.1 Host端计算流程

同样的，后向的Host端主要是准备Index映射所需要的数据，包括输入、输出张量的维数、每一维的尺寸、每一维的步长等。在这之后进行Kernel Launch即可。

## 2.2.2 Device端计算流程

Device端计算流程主要为以ElementWise方式寻找in_idx和out_idx之间的映射关系。由于后向属于ReductionSum过程，因此可调用Paddle封装好的CUDA工具，快速求取Block内数据之和，然后将结果赋值给输出张量的对应位置。

``````c++
int in_idx = blockIdx.x, out_idx;
T val = 0;

for(int i = threadIdx.x; i < acc_N; i += blockDim.x){
    out_idx = f(in_idx, i);
    tmp += out_grad[out_idx];
}

__syncthreads();
T result = funcs::BlockReduceSum<T>(val, FULL_MASK);

if(threadIdx.x == 0) in_grad[in_idx] = result;

``````

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)


# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2023-03-05 |
| 2 | 完成优化设计文档  | 2023-03-05 |
| 3 | expand_as优化实现  | 2023-03-10 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2023-03-15 |



# 5 影响面

需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响。


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
