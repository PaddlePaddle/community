# p_norm_grad OP性能优化设计文档


| 基本信息                                                     | 内容                                        |
| ------------------------------------------------------------ | ------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | zbt78                                   |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-05-31                                  |
| 版本号                                                       | V1.0                                        |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                               |
| 文件名                                                       | 20230531_p_norm_grad_op_optimization.md<br> |


# 1 背景与意义

目前 Paddle 内 p_norm_grad 算子 GPU 计算采用了 CUDA Kernel 与 Eigen 混合的模式，用现有的 Reduce OP 等取代 Eigen 可以提升计算性能，减少数据 HtoD 拷贝等开销。

## 1.1 飞桨现状

当前性能如下表(基于PaddlePaddle　develop分支)：

| Case No. | device | input_shape |input_type|porder|axis|  keepdim | Paddle Perf(ms) |
|---|---|---|---|---|---|---|---|
| 1 | GTX 960 | [300L, 128L, 128L] |float32| fro | -1 |False| 4.2505|
| 2 | GTX 960| [300L, 128L, 128L] |float32| 3.0 | -1 |False| 4.2476| 
| 3 | GTX 960 | [300L, 128L, 128L] |float32| 3.0 | -1 |True| 4.2596|
| 4 | GTX 960 | [300L, 128L, 128L] |float16| 3.0 | -1 |False| 4.3653| 


 [API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/norm_cn.html#norm)

## 1.2 业内方案调研

Pytorch对p_norm_grad算子的实现基于GPU计算。整体性能如下：

| Case No. | device | input_shape |input_type|porder|axis|  keepdim | Pytorch Perf(ms) |
|---|---|---|---|---|---|---|---|
| 1 | GTX 960 | [300L, 128L, 128L] |float32| fro | -1 |False| 1.6047|
| 2 | GTX 960| [300L, 128L, 128L] |float32| 3.0 | -1 |False| 2.4320| 
| 3 | GTX 960 | [300L, 128L, 128L] |float32| 3.0 | -1 |True| 2.4957|
| 4 | GTX 960 | [300L, 128L, 128L] |float16| 3.0 | -1 |False| 2.2846| 

 [Pytorch源码](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistanceKernel.cu)

## 1.3 对比分析

- 整体性能Pytorch优于飞桨，主要原因是飞桨的算子是基于Eigen实现的
- Pytorch针对参数`porder`的不同情况做了特殊处理

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

- 分类讨论，用现有的`BroadcastKernel`或`ElementWiseKernel`取代Eigen
- 注意与调用Eigen不同的是，需要对keepdim进行处理

## 2.2 Host端计算流程

- 根据`out.numel()`、`porder`、`keepdim`分类讨论
- `BroadcastKernel`或`ElementWiseKernel`内部已经含有了向量化读取、写入与1D线程配置

## 2.4 Device端计算流程

- 根据公式
$$dx = \frac {abs(x)^p * sign(x)*broadcast(dy)}{broadcast((y+eps)^p)}$$
进行计算。需要注意的是，对于out为scalar的情况下，关于y与dy的计算部分也为scalar，可以简化计算

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2023.05.31 |
| 2 | 完成开发文档设计  | 2023.05.31 |
| 3 | 提交PR进行后续迭代 | 2023.05.30 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
