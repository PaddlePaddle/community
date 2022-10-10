Poisson OP性能优化设计文档

| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   |  Rayman的团队  |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-08-10 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20220810_poisson_op_optimization.md<br> |

 # 1 背景与意义

目前Paddle中的PoissonKernel是通过cuRAND库+for_range + Functor组合实现。在device端调用cuRAND库是目前已知效率最高的随机数产生方案，性能可以进一步提升的空间在于优化线程的构建分布过程，以达到更优的效果。

##  1.1 飞桨现状

对于此OP在目前飞桨框架（Develop分支）中的性能现状调研，表格形式列出[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)中各种case场景下的OP性能数据（Tesla P4）。
| Case No. | input_shape | data_type | Paddle Perf(s) |
|---|---|---|---|
| 1 | [16, 16, 16, 16] |float32|0.3478 | 
| 2 | [16, 35, 1500] |float32| 2.9794|
| 3 | [16, 16, 16, 16] |float64 |0.2866|

 ## 1.2 业内方案调研

调研测试pytorch目前产生poisson分布的性能。
| Case No. | input_shape | data_type | Paddle Perf(s) |Perf_over_percent(%)
|---|---|---|---|---|
| 1 | [16, 16, 16, 16] |float32|0.3117 | +10.38
| 2 | [16, 35, 1500] |float32| 2.6733|+10.27
| 3 | [16, 16, 16, 16] |float64 |0.3202|-11.72

 ## 1.3 对比分析
对比表格1和表格2中的数据，case1和case2情况下是处理float32数据，pytorch相比较paddle有10%左右的性能提升。而case3是处理float64的数据，pytorch相对于paddle有11.7%的性能下降。

 # 2 设计方案与性能预期

 ## 2.1 关键模块与性能提升点
 + paddle于pytorch的性能差距在10%左右，差距并不太大，通过源码分析同样都是用了cuRAND函数库，因此试图改进poisson分布数值产生的方式以达到质的飞跃并不可取。
 + 性能提升关键在点在于优化Poisson_kernel.cu的host端代码，通过优化GPU上grid， block数量以寻找到更优参数以获得超过7%的提升。

##  2.2 Host端计算流程
在方案设计阶段有两种实验思路；
1. 方案一：通过paddle已实现的gpu_launch_config.h中GetGpuLaunchConfig1D方法获得较优的参数配置。该方案经过测试在float32数据上有5%左右的性能提升，float64数据上有10%左右的性能下降。故不作为首选方案。
2. 方案二：通过手动测试在该场景下更优的配置参数，BlockSize性能较优的取值通常为[128, 256,512]。对这三者进行实验并测试性能，结果显示是用一维Grid，且BlockSize=256时，在不同测试用例，不同测试环境中均有大幅性能提升。

 ## 2.3 Device端计算流程

保持原有逻辑，使用cuRAND中curand_poisson方法实现。

 ## 3 测试和验收的考量

实验环境1：Tesla P4
| Case No. | input_shape | data_type | Paddle_modify Perf(s) |Perf_over_paddle_origin(%)|Perf_over_pytorch(%)
|---|---|---|---|---|---|
| 1 | [16, 16, 16, 16] |float32|0.2205 | +36.62|+29.27
| 2 | [16, 35, 1500] |float32| 2.044|+31.40|+23.54
| 3 | [16, 16, 16, 16] |float64 |0.2159|+24.68|+32.57

 # 4 可行性分析和排期规划

已完成开发和测试，待优化代码规范后提交PR

预计提交时间：8.15


#  5 影响面

对其他模块没有影响。


 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
