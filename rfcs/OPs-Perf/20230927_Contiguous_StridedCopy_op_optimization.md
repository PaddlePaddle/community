# Lerp OP性能优化设计文档


| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | WintersMontagne10335   |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-09-27 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20230927_Contiguous_StridedCopy_op_optimization.md<br> |


# 1 背景与意义
目前ContiguousKernel、StridedCopyKernel两个 kernel 都是通过 numel index 计算数据偏移地址，需要一个 for 循环做计算，计算偏移地址效率低，导致 kernel 性能差。

## 1.1 飞桨现状

OP Benchmark暂无这两个op的测试文件。待后续补充测试数据。

## 1.2 业内方案调研

后续补充。

## 1.3 对比分析

后续补充。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

- 依靠线程配置信息，减少除法与取余操作。目前Paddle通过 numel index 计算数据偏移地址，会有大量除法与取余操作。可以利用线程配置中Grid与Block的6个参数（受硬件支持），减少除法与取余操作
- 去除依赖。由于kMaxRank为9，对于rank大于6的情况下，还是需要计算部分偏移地址。目前Paddle中的实现存在依赖关系。下一个循环只能等本循环的index_tmp计算完毕后才能进行，但其实各个循环间的计算从逻辑实现上来说是独立的，完全可以并行计算。本方法，与上一个优化方法搭配，可以大幅缩短运行时间
- 预处理访存偏移量（可选优化点）。借鉴MegEngine卷积算子预处理访存偏移量的优化思路
- 改变访存（可选优化点）。《CUDA_C优化详解》中提到，“应尽可能避免非单位跨度的全局内存访问”，对于stride比较特殊的情况，可以优化访存

## 2.2 Host端计算流程

- 依据input.dims()做线程配置

## 2.3 Device端计算流程

- 消除依赖

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)

# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2023-09-27 |
| 2 | 完成开发文档设计  | 2023-09-27 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2023-09-28 |


# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。

# 名词解释



# 附件及参考资料
[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
