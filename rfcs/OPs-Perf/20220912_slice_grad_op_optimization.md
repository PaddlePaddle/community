SliceGrad OP性能优化设计文档

| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   |  今天用GPU了吗的团队  |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-09-12 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20220912_slice_grad_op_optimization.md<br> |

 # 1 背景与意义

目前 Paddle 内 slice_grad 算子的 GPU 实现采用 Eigen 组合的模式，缺少 GPU Kernel，性能相对不足。

##  1.1 飞桨现状

目前的实现有一定的性能优化空间，可以加入一些性能优化的技巧。当前backward性能如下表：

| Case No. | device | input_shape | input_type | Paddle Perf(ms) |
|---|---|---|---|---|
| 1 | RTX 3070 |  |  |  | 
| 2 | RTX 3070 | |  |  |

 ## 1.2 业内方案调研

调研测试pytorch目前计算LerpGard的性能。
| Case No. | device | input_shape | input_type | Pytorch Perf(ms) |
|---|---|---|---|---|
| 1 | RTX 3070 |  |  |   | 
| 2 | RTX 3070 |  |  |  |

 ## 1.3 对比分析
目前Paddle与Pytorch的API设计方案几乎相同，两种case下测试pytorch性能更优，理论上可以通过向量化读取和写入等手段进行优化，进一步提升算子性能。

 # 2 设计方案与性能预期

 ## 2.1 关键模块与性能提升点
   将pad函数从eigen模式迁移到CUDA实现中，并对dims的计算进行优化


##  2.2 Host端计算流程
    通过gpu_launch_config.h中的线程配置方法配置1D线程。

 ## 2.3 Device端计算流程
    先对pad函数进行GPU实现，然后将dims移植到GPU计算，加快并行化访存


 ## 3 测试和验收的考量

通过单测以及Benchmark的速度测试

 # 4 可行性分析和排期规划

9.25完成开发及测试


#  5 影响面

可能会影响到Pad算子的GPU实现。不过初期仅将影响范围先知道Slice_grad 算子中，不会影响到其他算子Eigen对Pad的调用


 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
