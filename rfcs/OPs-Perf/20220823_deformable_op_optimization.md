# deformable_conv OP性能优化设计文档

| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   |  Rayman的团队  |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-08-23 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20220823_deformable_op_optimization.md<br> |

 # 1 背景与意义

目前Paddle中的Deformable_conv的GPU版是使用cuBlas+CUDA kernel实现的，kernel实现方式与论文原作者的实现类似是将CPU的kernel迁移到了GPU上，对于CUDA代码未进行针对性优化。

##  1.1 飞桨现状

对于此OP在目前飞桨框架（Develop分支）中的性能现状调研，表格形式列出[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)中各种case场景下的OP性能数据（Tesla P4）。

### 时间分析
通过benchmark运行deformable_conv前向的测试，获得如下GPU Activities时间
|No.| Time (%)| Time | Calls | Name | 
|---|---|---|---|---|
|1|64.20%|2.55725s|6000|maxwell_sgemm_128x64_nn|
|2|35.68%|1.42121s|6000|phi::funcs::ModulatedDeformableIm2colGpuKernel|

API时间：
|No.| Time (%)| Time | Calls | Name | 
|---|---|---|---|---|
|1|97.00%|3.90030s|1000|cudaDeviceSynchronize|
|2|1.87%|75.183ms|12000|cudaLaunchKernel|

得到如下性能表
| Case No. | input_shape | offset | weight | mask | data_type | Paddle Perf(s) |
|---|---|---|---|---|---|---| 
| 1 | [6, 512, 19, 34] | [6, 18, 19, 34] | [256, 512, 3, 3] | [6, 9, 19, 34] |float32| 3.99924 | 

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
   + 优化点1: 通过优化grid， block数量寻找更优配置;
   + 优化点2: 将deformable_conv_kernel_impl中计算像素和权重乘积的循环迁移到ModulatedDeformableIm2colGpuKernel中，将col_buffer的并行计算和output_3d的计算整合，减少部分搬运开销;
   + 优化点3: 将deformable_conv_kernel_impl中的（batch_size / im2col_step）次循环并行化，目前用循环的方式im2col_step完成后才能进行下一个step，等待时间是无必要的;
   + 优化点4: 单独优化deformable_conv_kernel_impl中计算像素和权重乘积的循环;
   + 优化点5: 优化deformable_conv_functor中ModulatedDeformableIm2colGpuKernel内的两层循环。
  
 根据kernels运行时间分析：65%时间消耗在No.5上，而cuBlas本身的实现难以有较大优化空间，所以单独优化两个kernel较难实现目标。

 根据CUDA API时间分析：97%的时间是在进行Device同步。
 
 通过nvvp查看GPU执行情况发现：总体来看运行过程是串行的，每个im2col结束后执行gemm，然后再进行下一个im2col，gemm的过程。


##  2.2 Host / Device 端计算流程
首先用伪代码描述目前的计算逻辑
```
deformable_conv_kernel_impl.h
  #数据准备
  for(i=0;i < batch_size / im2col_step; ++i)
    #输入该step的input_step，offset_step等
    col_buffer_step = ModulatedDeformableIm2col(input_step, offset_step, ....)
    #输出col_buffer
    for (g; g < groups; ++g)
    # 输入weight, col_buffer
        output_step = blas.MatMul(weight, col_buffer_step, ....)
  # 输出output

deformable_conv_functor.cu
   #输入input，offset
      col_buffer = Im2col_kernel(input, offset)
  # 输出col_buffer
```

1. 针对优化点1: 考虑通过paddle已实现的gpu_launch_config.h中GetGpuLaunchConfig1D方法获得较优的参数配置，或手动对BlockSize的不同大小进行性能测试验证（可能有一定优化空间）
   
2. 针对优化点2: ModulatedDeformableIm2colGpuKernel的Host端多接入两个参数，Device端计算完成col_buffer后继续计算output（可能有较大优化空间）


```
deformable_conv_kernel_impl.h
  #数据准备
  for (i; i < batch_size/ im2col_step; ++i)
      # 输入该step的input_step，offset_step, weight等
	    output_step = ModulatedDeformableIm2col(input_step, offset_step, ...)
  # 输出output

deformable_conv_functor.cu
   #输入input，offset
    col_offset = Im2col_kernel(input, offset)
    # 获得col_buffer
    for (g; g < groups; ++g) 		            
      output = blas.MatMul(weight, col_buffer)
   # 输出output

目的：减少col_buffer从device同步回host，再到device过程
```

3. 针对优化点3: 将整个im2col_step的过程并行化形成新的kernel，包含im2col和gemm两个步骤(可能有较大优化空间)
```
deformable_conv_kernel_impl.h
  #数据准备
  # 输入所有的input，offset等
    output = ModulatedDeformableIm2col(input, offset)
  # 输出output           


deformable_conv_functor.cu
   # 设置二维grid，每个kernel获取自己对应的input，offset，完成计算
      col_offset = Im2col_kernel(input, offset)
      # 获得col_buffer
      for (g; g < groups; ++g) 		            
          output = blas.MatMul(col_buffer, weight)
   # 输出output


目的：减少数据搬运同时增强并行
```

4. 针对优化点4: 乘积继续使用blas.MatMul实现，循环可以进行展开或实现新的GPU kernel尝试并行（benchmark中循环次数为1，优化空间较小）

5. 针对优化点5: Im2colGpuKernel中的两层循环可以尝试展开或使用实现 Child kernel将循环并行执行。（探索，不确定优化可行性）

 ## 3 测试和验收的考量

通过优化点1的测试，已经将ModulatedDeformableIm2colGpuKernel执行速度提升25%。

优化点2和优化点3可以提高整体的并行程度，尚无法推测具体可以提升多少，根据优化点1结果预计会使整体运行提高超过25%。

 # 4 可行性分析和排期规划

8.23～8.26测试优化点1

8.26～9.20测试优化点2～5


#  5 影响面

对其他模块没有影响。


 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
