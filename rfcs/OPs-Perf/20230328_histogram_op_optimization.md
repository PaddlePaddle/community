# Histogram OP性能优化设计文档


| 基本信息     | 内容                                      |
| ------------ | ----------------------------------------- |
| 提交作者     | zerorains                                 |
| 提交时间     | 2023-03-28                                |
| 版本号       | V1.0                                      |
| 依赖飞桨版本 | PaddleDevelop                             |
| 文件名       | 20230328_histogram_op_optimization.md<br> |


# 1 背景与意义

目前Paddle中的Histogram算子在GPU上自主进行了CUDA内核编程，但是在计算分区边界时使用了Eigen的操作过程，存在一定的优化空间。

## 1.1 飞桨现状

当前Paddle采用自主编写的CUDA Kernel执行Histogram的核心计算部分，但是在确定直方图边界时使用Eigen进行计算，当前性能如下表(基于PaddlePaddle　develop分支)：


| Case No. | device| input_shape | input_type | bins | min | max |old Paddle Perf(ms) |
|---|---|---|---|---|---|---|---|
| 1 | Tesla V100 | [16, 64] | int32 | 100 | 0 | 0 |    0.09403 |
| 2 | Tesla V100 | [16, 64] | int64 | 100 | 0    | 0 | 0.13624 |
| 3 | Tesla V100 | [16, 64] | float32 | 100 | 0 | 0 |  0.01889 |

API文档：[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/histogram_cn.html#histogram](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/histogram_cn.html#histogram)

## 1.2 业内方案调研


Pytorch对于[Histogram算子的实现](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/SummaryOps.cu#L55)也是采用了CUDA内核编程的方式，结合CUDA编程中的共享内存来存储最终的直方图结果，从而加快GPU中并行计算的速度。Paddle对于Histogram算子的实现，也是采用相同的策略进行的。两者之间的差距在于边界的确定上，Paddle使用Eigen进行边界计算使得性能稍微低于pytorch。

| Case No. | device| input_shape | input_type | bins | min | max |Pytorch Perf(ms) |
|---|---|---|---|---|---|---|---|
| 1 | Tesla V100 | [16, 64] | int32   | 100 | 0 | 0 | 0.02255 |
| 2 | Tesla V100 | [16, 64] | int64   | 100 | 0 | 0 | 0.03424 |
| 3 | Tesla V100 | [16, 64] | float32 | 100 | 0 | 0 | 0.02250 |

## 1.3 对比分析

在Benchmark中，对现在develop版本的Paddle进行GPU计算分析。在分析结果中，总共有超过90%的GPU计算时间使用在Eigen的计算中。在对源码进行分析后，Eigen的计算使用在确定直方图边界上，并不属于`Histogram`算子的核心计算内容。这显然是不合理的。同时参考Pytorch的源码之后，Paddle和Pytorch在`Histogram`算子的源码实现上基本一致，因此使用新的方法替换Eigen的计算是`Histogram`算子在GPU的计算性能提升的关键。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

关键是使用__global__ kernel的方式实现了`KernelMinMax`，加速`Histogram`确定直方图边界的计算部分，从而提高`Histogram`算子在GPU上的计算性能。预期能够平均提升2倍以上。

## 2.2 Host端计算流程

Host端需要为`Histogram`算子提供两个部分的数据，第一个部分是为确定直方图边界的`__global__ kernel`提供相应的输出变量，第二部分是将确定好的边界从Device端移动到Host端，为`Histogram`的核心计算部分提供边界信息。

具体来说，首先就是首先创建一个大小为2的DenseTensor，然后分配内存，将返回的指针作为输出，传入到手动编写的`__global__ kernel`中。然后将边界输出传递给核心计算的`HistogramKernel`中即可。

## 2.4 Device端计算流程

Device端则是参考了`KernelHistogram`的写法，借助共享内存和`CUDA_KERNEL_LOOP`函数，实现了`KernelMinMax`，在CUDA Kernel内部并采取`phi::CudaAtomicMin`和`phi::CudaAtomicMax`函数实现边界值的寻找。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)

完成Tile OP 开发后，新的Paddle与旧的Paddle性能对比效果如下：

| Case No. | device| input_shape | input_type | bins | min | max |Paddle Perf(ms) |old Paddle Perf(ms) |diff|
|---|---|---|---|---|---|---|---|---|---|
| 1 | Tesla V100 | [16, 64] | int32   | 100 | 0 | 0 | 0.01220 |0.09403|faster than 670.74% |
| 2 | Tesla V100 | [16, 64] | int64   | 100 | 0 | 0 | 0.01515 |0.13624|faster than 799.27%|
| 3 | Tesla V100 | [16, 64] | float32 | 100 | 0 | 0 | 0.01501 |0.01889|faster than 25.85%|

新的Paddle与Pytorch性能对比效果如下，达到了预期性能提升效果：

| Case No. | device| input_shape | input_type | bins | min | max |Paddle Perf(ms) |Pytorch Perf(ms) |diff|
|---|---|---|---|---|---|---|---|---|---|
| 1 | Tesla V100 | [16, 64] | int32   | 100 | 0 | 0 | 0.01220 |0.02255|faster than 84.84%|
| 2 | Tesla V100 | [16, 64] | int64   | 100 | 0 | 0 | 0.01515 |0.03424|faster than 126.01%|
| 3 | Tesla V100 | [16, 64] | float32 | 100 | 0 | 0 | 0.01501 |0.02250|faster than 49.90%|

针对三种不同的Case，优化后性能有不同程度的提升。   

# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 3.25~3.28 |
| 2 | 完成开发文档设计  | 3.28~3.29 |
| 3 | 提交PR进行后续迭代 | 3.29~活动结束 |


# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释



# 附件及参考资料
[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)

