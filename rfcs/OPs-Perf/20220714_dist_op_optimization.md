# dist OP性能优化设计文档


| 基本信息                                                     | 内容                                   |
| ------------------------------------------------------------ |--------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-07-14                           |
| 版本号                                                       | V1.0                                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                        |
| 文件名                                                       | 20220714_dist_op_optimization.md<br> |


# 1 背景与意义

目前Paddle中的Dist算子已通过基于Kernel Primitive API实现的PNormKernel达到很不错的性能效果。 
待挖掘的性能提升方面可能可以基于原生的自定义算子实现。

## 1.1 飞桨现状

当前性能如下表(基于ＰaddleＰaddle　develop分支)：

| Case No. | input_shape |　ｐ | Ｐaddle Perf(ms) |
|---|---|---|---|
| 0 | [1000,1000] | 2.0 | 0.2338 | 
| 1 | [1000,1000] | inf　| 0.1843 | 
| 2 | [1000,1000] |　0 | 0.1586 | 

三种Case都基于形状[1000,1000]的输入，只是p的取值不一样，分别是２.0, inf, 0。
当前API设计文档: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/dist_cn.html#dist

## 1.2 业内方案调研

Pytorch中对dist算子的实现基于GPU计算,  整体性能如下(基于Ｐytorch　v1.12)：

| Case No. | input_shape |　ｐ | Ｐytorch Perf(ms) |
|---|---|---|---|
| 0 | [1000,1000] |　2.0  |  0.2492 | 
| 1 | [1000,1000] | inf　|  0.2134 | 
| 2 | [1000,1000] |　0 | 0.1586 | 

## 1.3 对比分析

目前Paddle与Pytorch的API设计方案几乎相同，3种case测试发现均优于Pytorch的实现。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

对kps:reduce进行改写，将中间步骤合并到一个kernel执行，预计提升1.3倍以上。

## 2.2 Host端计算流程

通过broadcast对齐输入的tensor形状。

## 2.4 Device端计算流程

设备端通过kps::ReadData和kps::WriteData对数据进行读写，再对每对值进行dist运算。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2022-07-11 |
| 2 | 完成开发文档设计  | 2022-07-17 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2022-07-24 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)


