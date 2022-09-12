# matrix_rank OP性能优化设计文档

| 基本信息                 | 内容     |
| ---- | -------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   |  OccupyMars2025  |                                         
| 提交时间<input type="checkbox" class="rowselector hidden"> | 2022-09-12 |                                                
| 版本号     | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| develop|                      
| 文件名    | 20220912_matrix_rank_op_optimization.md <br> |

 # 1 背景与意义

##  1.1 飞桨现状

现状：目前 Paddle 内 matrix_rank 算子采用第三方库组合实现，性能不足；
目标：请优化计算实现，为 Paddle 优化 matrix_rank op 在 GPU 上的计算性能，性能至少提升3倍，针对性能差的case，性能提升120+倍。

 ## 1.2 业内方案调研


 ## 1.3 对比分析


 # 2 设计方案与性能预期
设计文档：提 PR 至 community repo 的 rfcs/OPs-Perf 目录；
C++ 及 GPU kernel 实现代码：提 PR 至 paddle/phi/kernels/gpu/matrix_rank_kernel.cu 目录；
在代码 PR 中提交 OP 的性能数据变化表格，代码 PR 格式规范见 示例。

 ## 2.1 关键模块与性能提升点


##  2.2 Host端计算流程


 ## 2.3 Device端计算流程


 ## 3 测试和验收的考量

通过单测以及Benchmark的速度测试

 # 4 可行性分析和排期规划

2022年10月1日-10月15日完成开发及测试


#  5 影响面

对其他模块没有影响。


 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
