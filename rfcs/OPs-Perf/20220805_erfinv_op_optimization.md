# erfinv OP性能优化设计文档


| 基本信息                                                     | 内容                                   |
| ------------------------------------------------------------ |--------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-08-05                           |
| 版本号                                                       | V1.0                                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                        |
| 文件名                                                       | 20220805_erfinv_optimization.md<br> |


# 1 背景与意义

目前Paddle中的Erfinv是通过Eigen组合实现，没有用到一些性能优化的技巧，存在性能优化的空间。

## 1.1 飞桨现状

当前性能如下表(基于PaddlePaddle　develop分支)：

目前的实现有一定的性能优化空间，可以加入一些性能优化的技巧。当前forward性能如下表：

| Case No. | device | input_shape | input_type | Paddle Perf(ms) |
|---|---|---|---|---|
| 1 | RTX 2070s | [-1L, 204800L] | float32 | 0.1438 | 
| 2 | RTX 2070s |[10L, 20L, 30L, 40L, 5L, 6L] | float64 | 8.6485 |

API文档 https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/erfinv_cn.html#erfinv

## 1.2 业内方案调研

Pytorch中对Erfinv算子的实现基于GPU计算,  forward整体性能如下(基于pytorch　v1.12)：

| Case No. | device | input_shape | input_type | Pytorch Perf(ms) |
|---|---|---|---|---|
| 1 | RTX 2070s | [-1L, 204800L] | float32 |  0.0677 | 
| 2 | RTX 2070s |[10L, 20L, 30L, 40L, 5L, 6L] | float64 | 2.7904 |

## 1.3 对比分析

目前Paddle与Pytorch的API设计方案几乎相同，两种case下测试pytorch性能更优，理论上可以通过向量化读取和写入等手段进行优化，进一步提升算子性能。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

通过使用飞桨内部的Elementwise Kernel来进行计算。通过向量化读取、向量化写入以及gpu_launch_config.h中的线程配置方法对算子进行优化, 使用cuda内置函数后预计比当前算子提升２倍以上。

## 2.2 Host端计算流程

通过gpu_launch_config.h中的线程配置方法配置1D线程。

## 2.4 Device端计算流程

设备端通过kps::ReadData和kps::WriteData对数据进行读写，再对每个值进行erfinv计算。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2022-08-05 |
| 2 | 完成开发文档设计  | 2022-08-08 |
| 3 | ndtri函数开发  | 2022-08-09 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2022-08-10 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)


