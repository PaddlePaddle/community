# argmin_argmax OP性能优化设计文档


| 基本信息                                                     | 内容                                   |
| ------------------------------------------------------------ |--------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-09-12                        |
| 版本号                                                       | V1.0                                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                        |
| 文件名                                                       | 20220912_argmin_max_op_optimization.md<br> |


# 1 背景与意义

目前Paddle中的argmin_argmax算子的GPU实现采用了Cub库实现，性能还有进一步提升的空间；

## 1.1 飞桨现状

当前性能如下表(基于PaddlePaddle　develop分支)：

| Case No. | input_shape |　dtype | axis | Paddle Perf(ms) argmin | Paddle Perf(ms) argmmax |
|---|---|---|---|---|---|
| 0 | [-1L, 513L, 513L, 19L] | float32 | 3 | 15.0504 | 15.0504 | 
| 1 | [-1L, 513L, 513L, 19L] | float32 | 1 | 20.0625 | 20.0625 | 
| 2 | [1000L, 1000L] | float32 | -1 | 0.16095 | 0.16095 | 
| 3 | [1000L, 1000L] | float32 | 0 | 0.7225 | 0.7225 | 

当前API设计文档: 
https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmin_cn.html#argmin
https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/argmax_cn.html#argmax

## 1.2 业内方案调研

Pytorch中对argmin_argmax算子的实现基于GPU计算,  整体性能如下(基于Ｐytorch　v1.12)：

| Case No. | input_shape |　dtype | axis | Pytorh Perf(ms) argmin | Pytorh Perf(ms) argmmax |
|---|---|---|---|---|---|
| 0 | [-1L, 513L, 513L, 19L] | float32 | 3 | 10.426 | 10.426 | 
| 1 | [-1L, 513L, 513L, 19L] | float32 | 1 | 2.4442 | 2.4442 | 
| 2 | [1000L, 1000L] | float32 | -1 | 0.03902 | 0.03902 | 
| 3 | [1000L, 1000L] | float32 | 0 | 0.04725 | 0.04725 | 

## 1.3 对比分析

目前Paddle与Pytorch的API设计方案几乎相同, Paddle底层都使用了Cub库实现, Pytorch底层基于一维的gpu_reduce_kernel采用了reduce方案。
PaddlePaddle对block的设计更讲究，按照２的指数向上取最优的block配置。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

基于kps:reduce进行改写，优化grid和block配置，并尝试二维的线程配置策略进一步提升性能，预计性能平均至少提升4.5倍，以及能达到pytorch目前的性能。

## 2.2 Host端计算流程

通过broadcast对齐输入的tensor形状。

## 2.4 Device端计算流程

设备端通过kps::ReadData和kps::WriteData对数据进行读写，再对每对值进行argmin/argmax运算。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2022-09-12 |
| 2 | 完成开发文档设计  | 2022-09-12 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2022-09-25 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)

