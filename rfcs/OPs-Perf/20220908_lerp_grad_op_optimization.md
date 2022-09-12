LerpGrad OP性能优化设计文档

| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   |  Rayman的团队  |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-09-08 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20220908_lerp_grad_op_optimization.md<br> |

 # 1 背景与意义

目前Paddle中的LerpGradKernel是通过Eigen组合的模式实现的，没有单独的GPU kernel，部分case优于竞品，部分case性能不足。通过实现GPU kernel提升较差情况下的性能。

##  1.1 飞桨现状

对于此OP在目前飞桨框架（Develop分支）中的性能现状调研，表格形式列出[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)中各种case场景下的OP性能数据（Tesla P4）。
| Case No. | input_shape | data_type | Paddle Perf(us) |
|---|---|---|---|
| 1 | [16, 102400] |float32|391.93 | 
| 2 | [16, 204800] |float64| 875.49|

 ## 1.2 业内方案调研

调研测试pytorch目前计算LerpGard的性能。
| Case No. | input_shape | data_type | Torch Perf(s) |Perf_over_percent(%)
|---|---|---|---|---|
| 1 | [16, 102400] |float32| 228.93 | +41.5
| 2 | [16, 204800] |float64| 945.54 | -8%

 ## 1.3 对比分析
对比表格1和表格2中的数据，float32情况下pytorch速度要优于目前的实现，float64情况下若与目前实现。

 # 2 设计方案与性能预期

 ## 2.1 关键模块与性能提升点
   目前实现没有充分利用GPU并行性能。
   + 首先该操作是ElementWise的操作，可以使用ElementWise模版提高速度
   + 其次目前对于x的导数Dx、对于y的导数Dy目前是串行算的，如下式重复计算了 W * Dout的乘法，但是通过推导有Dx = Dout - Dy的形式，可以直接在一次操作中通过减法实现从而提高速度

   $$ D_y = W * D_{out}$$
   $$ D_x = (1 - W) * D_{out}$$


##  2.2 Host端计算流程
    通过BroadCast获得完整的数据，调用Elementwise模版

 ## 2.3 Device端计算流程
    根据 Dy = W * Dout, Dx = Dout - Dy来计算导数


（尝试点： 由于x, y, w是需要BroadCast到同样维度的，也可以尝试将低维度的数据放进共享内存中调用，而不事先BroadCast数据）
 ## 3 测试和验收的考量

通过单测以及Benchmark的速度测试

 # 4 可行性分析和排期规划

9.25完成开发及测试


#  5 影响面

对其他模块没有影响。


 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
