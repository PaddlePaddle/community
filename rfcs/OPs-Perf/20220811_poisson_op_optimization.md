# poisson OP性能优化设计文档


| 基本信息                                                     | 内容                                   |
| ------------------------------------------------------------ |--------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-08-11                           |
| 版本号                                                       | V1.0                                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                        |
| 文件名                                                       | 20220811_poisson_optimization.md<br> |


# 1 背景与意义

目前Paddle中的Poisson op是实现采用 for_range() 组合的模式，存在性能优化的空间。

## 1.1 飞桨现状

当前性能如下表(基于PaddlePaddle　develop分支)：

目前的实现有一定的性能优化空间，可以加入一些性能优化的技巧。当前forward性能如下表：

| Case No. | device | input_shape | input_type | Paddle Perf(ms) |
|---|---|---|---|---|
| 0 | RTX 2070s | [-1L, 16L, 16L, 16L] | float32 | 0.15552 | 
| 1 | RTX 2070s | [-1L, 35L, 1500L] | float32| 1.35704 |
| 2 | RTX 2070s | [-1L, 16L, 16L, 16L] | float64| 0.15577 |

API文档 https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/poisson_cn.html#poisson

## 1.2 业内方案调研

Pytorch中对Poisson算子的实现基于GPU计算,  forward整体性能如下(基于pytorch　v1.12)：

| Case No. | device | input_shape | input_type | Pytorch Perf(ms) |
|---|---|---|---|---|
| 0 | RTX 2070s | [-1L, 16L, 16L, 16L] | float32 | 0.14175 | 
| 1 | RTX 2070s | [-1L, 35L, 1500L] | float32| 1.19954 |
| 2 | RTX 2070s | [-1L, 16L, 16L, 16L] | float64| 0.15137 |

## 1.3 对比分析

目前Paddle与Pytorch的API设计方案几乎相同，case 1的情况下有明显的差距，另外两种case性能差距不大。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

通过使用飞桨内部的gpu_launch_config.h中的线程配置方法对算子进行优化，预计提升%7以上。

## 2.2 Host端计算流程

通过gpu_launch_config.h中的线程配置方法配置1D或2D线程。

## 2.4 Device端计算流程

算子的逻辑通过PoissonCudaFunctor实现。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2022-08-11 |
| 2 | 完成开发文档设计  | 2022-08-11 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2022-08-15 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)



