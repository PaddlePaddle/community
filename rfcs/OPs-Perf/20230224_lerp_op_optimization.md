# lerp OP性能优化设计文档


| 基本信息                                                     | 内容                                   |
| ------------------------------------------------------------ |--------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | wfs2010                            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-24                           |
| 版本号                                                       | V1.0                                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                        |
| 文件名                                                       | 20220224_lerp_op_optimization.md<br> |


# 1 背景与意义

目前Paddle中的算子已通过基于第三方库Eigen组合实现达到很不错的性能效果,Paddle中暂时没有GPU版本的lerp OP，导致训练过程中采用第CPU执行对应计算，性能明显存在瓶颈，影响了模型性能。

## 1.1 飞桨现状

Paddle中暂时没有lerp OP的GPU实现，需要实现一个GPU版本的lerp OP.当前基于第三方库的性能如下表(基于PaddleＰaddle　develop分支)：

| Case No. | device |　input_shape | input_x_type |input_y_type | weight_type | Paddle Perf(ms) |
|---|---|---|---| ---| ---| ---|
| 0 | rtx3090 | [1000,1000] | float32 | float32 | float32 | 0.5183|
| 1 | rtx3090 | [5100,5100] | float32 | float32 | float32 | 0.9386|

两种Case都基于不同形状的输入。
当前API设计文档: https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/lerp_cn.html#lerp

## 1.2 业内方案调研

Pytorch中对lerp算子的实现基于GPU计算,  整体性能如下(基于Ｐytorch　v1.13.1+cu117)：

| Case No. | device |　input_shape | input_x_type |input_y_type | weight_type | Pytorch Perf(ms) |
|---|---|---|---| ---| ---| ---|
| 0 | rtx3090 | [1000,1000] | float32 | float32 | float32 | 0.1413|
| 1 | rtx3090 | [5100,5100] | float32 | float32 | float32 | 0.3433|


## 1.3 对比分析

目前Paddle与Pytorch的API设计方案几乎相同，测试发现均优于Paddle的实现。由于Paddle中没有GPU实现，考虑到lerp的计算多用于多维数据，因此决定采用多维线程设置，减少数据索引部分的计算量，2维线程设置方案如下：
通过grid和block的优化配置性能明显优于paddle，在性能上提升超过原始实现。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

使用cuda重写lerp的实现，使用cuda内置函数后预计比当前算子提升%20以上。

## 2.2 Host端计算流程

通过gpu_launch_config.h中的线程配置方法配置线程

## 2.4 Device端计算流程

设备端通过kps::ReadData和kps::WriteData对数据进行读写，再对每对值进行lerp运算。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间       |
|---|---|------------|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2023-02-23 |
| 2 | 完成开发文档设计  | 2023-02-23 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2023-02-26 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)


