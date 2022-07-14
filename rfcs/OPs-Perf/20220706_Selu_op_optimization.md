# Selu OP性能优化设计文档


| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">   | carryyu                                               |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-07-06                                                   |
| 版本号                                                       | V1.0                                   |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                      |
| 文件名                                                       | 20220706_Selu_op_optimization.md<br> |

# 1 背景与意义

目前Paddle中的Selu是通过Eigen组合实现，没有用到一些性能优化的技巧，存在性能优化的空间。

## 1.1 飞桨现状

目前的实现有一定的性能优化空间，可以加入一些性能优化的技巧。当前性能如下表：
| Case No. | device | input_shape | input_type | Paddle Perf(ms) |
|---|---|---|---|---|
| 1 | Tesla T4 | [8, 1024, 3072] | float32 | 0.9122 | 
| 1 | Tesla T4 | [8, 1024, 3072] | float64 | 5.2592 |

## 1.2 业内方案调研

Pytorch中对应`paddle.nn.functional.selu` 的Api为 `torch.nn.functional.selu`。调研发现Pytorch中采用的是`SeluKernel` Kernel完成该OP的GPU实现。PyTorch采用的方案是1维线程设置完成整体计算，整体性能如下：
| Case No. | device | input_shape | input_type | Pytorch Perf(ms) |
|---|---|---|---|---|
| 1 | Tesla T4 | [8, 1024, 3072] | float32 | 0.8349 | 
| 1 | Tesla T4 | [8, 1024, 3072] | float64 | 5.4939 |

## 1.3 对比分析

目前Paddle与Pytorch的方案几乎相同，但理论上可以通过向量化读取和写入等手段进行优化，进一步提升算子性能。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

通过使用飞桨内部的Elementwise Kernel来进行计算。通过向量化读取、向量化写入以及gpu_launch_config.h中的线程配置方法对算子进行优化，预计提升5%。

## 2.2 Host端计算流程

通过gpu_launch_config.h中的线程配置方法配置1D线程。

## 2.4 Device端计算流程

设备端通过kps::ReadData和kps::WriteData对数据进行读写，再对每个值进行selu计算。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)
完成优化后，Paddle与优化前的Paddle的性能对比效果如下，达到了预期性能提升效果（提升5%）：
| Case No. | device | input_shape | input_type | Paddle Perf(ms) | Old-Paddle Perf(ms) | diff |
|---|---|---|---|---|---|---|
| 1 | Tesla T4 | [8, 1024, 3072] | float32 | 0.8277 | 0.9122 | faster than 9.26% |
| 1 | Tesla T4 | [8, 1024, 3072] | float64 | 4.5655 | 5.2592 | faster than 13.19% |

完成优化后，Paddle与Pytorch的性能对比效果如下，在fp32情况下基本与Pytorch持平，在fp64情况下提升较大 ：
| Case No. | device | input_shape | input_type | Paddle Perf(ms) | Pytorch Perf(ms) | diff |
|---|---|---|---|---|---|---|
| 1 | Tesla T4 | [8, 1024, 3072] | float32 | 0.8277 | 0.8349 | faster than 0.86% |
| 1 | Tesla T4 | [8, 1024, 3072] | float64 | 4.5655 | 5.4939 | faster than 16.89% |

# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2022-07-06 |
| 2 | 完成开发文档设计  | 2022-07-14 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2022-07-17 |

# 5 影响面

需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响。

# 名词解释

# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
