# Tile OP性能优化设计文档


| 基本信息     | 内容                                 |
| ------------ | ------------------------------------ |
| 提交作者     | zerorains                            |
| 提交时间     | 2023-03-19                           |
| 版本号       | V1.0                                 |
| 依赖飞桨版本 | PaddleDevelop                        |
| 文件名       | 20230319_tile_op_optimization.md<br> |


# 1 背景与意义

目前Paddle中的Tile算子在GPU和CPU的计算逻辑相同，没有编写对应的Cuda代码，存在一定优化空间

## 1.1 飞桨现状

当前实现没有自主进行CUDA编程，当前性能如下表(基于PaddlePaddle　develop分支)：


| Case No. | device|repeat_times | input_shape | input_type |old Paddle Perf(ms) |
|---|---|---|---|---|---|
| 1 | Tesla V100 |[1,10,128,128] | [16L,100L,2L,2L]| float32 | 10.1888 | 
| 2 | Tesla V100 |[1,10,128,128] | [16L,100L,2L,2L]| float16 | 16.7348 | 
| 3 | Tesla V100 |[4,1,807]     |  [32L,807L,1L] | float32 | 0.7381 | 
| 4 | Tesla V100 |[4,1,807]     |  [32L,807L,1L] | float16 | 0.9850 |



API文档：[https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tile_cn.html#tile](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tile_cn.html#tile)

## 1.2 业内方案调研


Pytorch中有`torch.tile`与`paddle.tile`对应，但是在pytorch中直接搜索`tile`对应的位置是在caffe2 namspace中的
[tile_op.cu](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/tile_op.cu)，pytorch应该是使用自己原生的`tile`方法，而不是caffe2中的，但是在aten中又找不到，经过查询[Paddle算子对照表](https://aistudio.baidu.com/aistudio/projectdetail/3464974)，了解到`paddle.tile`又可以和[torch.repeat](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/Repeat.cu)对应。

经过对两个代码的调研，在`tile_op.cu`中使用的是CUDA原生内核的编写方法，在用`__global__`标识的函数中编写执行逻辑，实现该OP在GPU中的计算。采用的是1维线程完成整体计算。`torch.repeat`中使用的也是CUDA原生内核的编写方法，在用`__global__`标识的函数中编写执行逻辑，实现该OP在GPU中的计算，采用的也是1维线程完成整体计算。

| Case No. | device|repeat_times | input_shape | input_type |Pytorch Perf(ms) |
|---|---|---|---|---|---|
| 1 | Tesla V100 |[1,10,128,128] | [16L,100L,2L,2L]| float32 | 8.0796 | 
| 2 | Tesla V100 |[1,10,128,128] | [16L,100L,2L,2L]| float16 | 7.7898 | 
| 3 | Tesla V100 |[4,1,807]     |  [32L,807L,1L] | float32 |   0.5342 | 
| 4 | Tesla V100 |[4,1,807]     |  [32L,807L,1L] | float16 |   0.3768 | 

## 1.3 对比分析

Paddle中是使用Eigen实现Tile的通用计算方式，并通过在.cu和.cc文件中直接对其进行注册完成计算的。而Pytorch则是采用1维线程完成整体计算。于是最简单的优化思路就是直接按照原本的逻辑编写对应的CUDA版本，但是考虑到Tile中的复制操作是可以使用`phi::funcs::BroadcastKernel`和`kps::IdentityFunctor<T>()`结合进行实现的，因此可以利用这两个方法替换复制的操作过程对其进行优化。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

关键是使用`phi::funcs::BroadcastKernel`与`kps::IdentityFunctor<T>()`的组合方式，加速`tile`执行中的复制操作，预计性能平均提升一倍。

## 2.2 Host端计算流程

`phi::funcs::BroadcastKernel`方法只能在一个维度进行扩展，因此在跟着`repeat_times`执行`repeat`的过程中，应该按顺序进行一维一维的拓展。下面是伪代码

```shell
init: vector<int> repeat_times, DenseTensor x, DenseTensor* output

vector<int> output_dims <- repeat_times // 记录每次braodcast之后的输出维度
for i=0 to repeat_times.size() do
  output_dims[i] <- output_dims[i] * repeat_times[i] // 本次复制后要输出的维度
  output->Resize(output_dims)                        // 改变输出维度
  // 这个方法不是直接这么用的，这里只是做个演示，目的是在第i个维度复制x，并将结果放到output上
  phi::funcs::BroadcastKernel(x,output,axis=i)
  x <- output // 将复制的结果赋值给x进行下一轮复制
end for
```

## 2.4 Device端计算流程

Device端则是按照Host端处理好的输入输出信息，调用`phi::funcs::BroadcastKernel`和`kps::IdentityFunctor<T>()`的组合进行复制即可。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)

完成Tile OP 开发后，新的Paddle与旧的Paddle性能对比效果如下：

| Case No. | device|repeat_times | input_shape | input_type |Paddle Perf(ms) |old Paddle Perf(ms) |diff |
|---|---|---|---|---|---|---|---|
| 1 | Tesla V100 |[1,10,128,128] | [16L,100L,2L,2L]| float32 | 5.1831 |10.1888|faster than 96.58%|
| 2 | Tesla V100 |[1,10,128,128] | [16L,100L,2L,2L]| float16 | 3.5461 |16.7348|faster than 372%|
| 3 | Tesla V100 |[4,1,807]     |  [32L, 807L, 1L] | float32 | 0.3885 |0.7381|faster than 89.99%|
| 4 | Tesla V100 |[4,1,807]     |  [32L, 807L, 1L] | float16 | 0.2465 |0.9850|faster than 300%|

新的Paddle与Pytorch性能对比效果如下，达到了预期性能提升效果：

| Case No. | device|repeat_times | input_shape | input_type |Paddle Perf(ms) |Pytorch Perf(ms) |diff |
|---|---|---|---|---|---|---|---|
| 1 | Tesla V100 |[1,10,128,128] | [16L,100L,2L,2L]| float32 | 5.1831 |8.0796|faster than 55.88%|
| 2 | Tesla V100 |[1,10,128,128] | [16L,100L,2L,2L]| float16 | 3.5461 |7.7898|faster than 120%|
| 3 | Tesla V100 |[4,1,807]     |  [32L, 807L, 1L] | float32 | 0.3885 |0.5342|faster than 37.50%|
| 4 | Tesla V100 |[4,1,807]     |  [32L, 807L, 1L] | float16 | 0.2465 |0.3768|faster than 52.86%|

针对四种不同的Case，优化后性能有不同程度的提升。   

# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 3.19~3.25 |
| 2 | 完成开发文档设计  | 3.25~3.26 |
| 3 | 提交PR进行后续迭代 | 3.26~活动结束 |


# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释



# 附件及参考资料
[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)

