Poisson OP性能优化设计文档

| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   |  Rayman的团队  |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-08-23 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20220823_deformable_op_optimization.md<br> |

 # 1 背景与意义

目前Paddle中的Deformable_conv的GPU版是使用cuBlas+CUDA kernel实现的，kernel实现方式与论文原作者的实现类似是讲CPU的kernel迁移到了GPU上，对于CUDA代码未进行针对性优化。

##  1.1 飞桨现状

对于此OP在目前飞桨框架（Develop分支）中的性能现状调研，表格形式列出[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)中各种case场景下的OP性能数据（Tesla P4）。

### 时间分析
通过benchmark运行deformable_conv的测试，其中执行了前向和后向的过程，需要对二者使用时间进行拆分, 下表列出核心的几部分。
|No.| Time (%)| Time | Calls | Name | 
|---|---|---|---|---|
|1 | 34.62% | 9.64260s | 6000 | phi::ModulatedDeformableCol2imCoordGpuKernel|
|2 | 25.90% | 7.21392s | 6000 | phi::ModulatedDeformableCol2imGpuKernel|
|3 | 9.98% | 2.78098s | 12000 | phi::funcs::ModulatedDeformableIm2colGpuKernel|
|4 | 9.36% | 2.60613s | 6000 | sgemm_128x128x8_TN|
|5 | 9.01% | 2.50930s | 6000 | maxwell_sgemm_128x64_nn|
|6 | 8.70% | 2.42293s | 6000 | sgemm_128x128x8_NT|

结合代码实现逻辑，分析得到：前向执行时进行的操作为6000次的No.3, 以及6000次的No.5, 其余均为后向执行时调用。故得到如下性能表

| Case No. | input_shape | offset | weight | mask | data_type | Paddle Perf(s) |
|---|---|---|---|---|---|---| 
| 1 | [6, 512, 19, 34] | [6, 18, 19, 34] | [256, 512, 3, 3] | [6, 9, 19, 34] |float32| 3.99662 | 

 ## 1.2 业内方案调研

调研测试pytorch目前执行Deformable conv的实现与Paddle接近，也是与论文原作者的实现方式类似，通过与1.1类似的分析方式得到下表。
| Case No. | input_shape | offset | weight | mask | data_type | Pytorch Perf(s) | Perf_over_percent(%)|
|---|---|---|---|---|---|---| --|
| 1 | [6, 512, 19, 34] | [6, 18, 19, 34] | [256, 512, 3, 3] | [6, 9, 19, 34] |float32| 5.7679 | -44%


 ## 1.3 对比分析
对比表格1和表格2中的数据，Pytorch的实现运行速度比Paddle慢。

 # 2 设计方案与性能预期

 ## 2.1 关键模块与性能提升点
 + 通过调研目前开源代码中与Deformable conv相关的实现都采用了统一的逻辑，即参考论文原作者的方式。
 + 可能的性能提升关键在点在于以下几方面：
   + 优化点1: 通过优化grid， block数量寻找更优配置。
   + 优化点2: 将deformable_conv_kernel_impl中计算像素和权重乘积的循环迁移到ModulatedDeformableIm2colGpuKernel中，将col_buffer的并行计算和output_3d的计算整合，减少部分搬运开销
   + 优化点3: 单独优化deformable_conv_kernel_impl中计算像素和权重乘积的循环；
   + 优化点4: 优化deformable_conv_functor中ModulatedDeformableIm2colGpuKernel内的两层循环。
  
 根据kernels运行时间分析：65%时间消耗在No.5上，而cuBlas本身的实现难以有较大优化空间，所以单独优化两个kernel较难实现目标

 根据CUDA API时间分析：63%时间消耗在同步，26%时间消耗在内存分配。通过减少需要线程同步的次数，降低数据在内存和CUDA间迁移应该可以较大程度优化。故优化点1和优化点2是重点考虑的对象。

##  2.2 Host / Device 端计算流程
1. 针对优化点1: 考虑通过paddle已实现的gpu_launch_config.h中GetGpuLaunchConfig1D方法获得较优的参数配置，或手动对BlockSize的不同大小进行性能测试验证（可能有一定优化空间）
   
2. 针对优化点2: ModulatedDeformableIm2colGpuKernel的Host端多接入两个参数，Device端计算完成col_buffer后继续计算output（可能有较大优化空间）

3. 针对优化点3: 乘积继续使用blas.MatMul实现，循环可以进行展开或实现新的GPU kernel尝试并行（benchmark中循环次数为1，优化空间较小）

4. 针对优化点3: Im2colGpuKernel中的两层循环可以尝试展开或使用实现 Child kernel将循环并行执行。（探索，不确定优化可行性）

 ## 3 测试和验收的考量

实现前向速度提升超过25%

 # 4 可行性分析和排期规划

8.23～8.26测试优化点1

8.26～9.15测试优化点2～4


#  5 影响面

对其他模块没有影响。


 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
