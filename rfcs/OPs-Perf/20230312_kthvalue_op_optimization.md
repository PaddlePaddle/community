# Kthvalue OP性能优化设计文档


| 基本信息                                                     | 内容                                   |
| ------------------------------------------------------------ |--------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-12                           |
| 版本号                                                       | V1.0                                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                        |
| 文件名                                                       | 20230312_kthvalue_op_optimization.md<br> |


# 1 背景与意义

目前 Paddle 内 kthvalue 算子 GPU 计算采用了cub库实现，性能仍有明显的提升空间。

## 1.1 飞桨现状

当前性能如下表(基于PaddlePaddle　develop分支)：

目前的实现有一定的性能优化空间，可以加入一些性能优化的技巧。当前forward性能如下表：

| Case No. | device | input_shape | input_type | k | Paddle Perf(ms) |
|---|---|---|---|---|---|
| 1 | RTX 2070s | [16L, 10000L] | float32 | 5 | 0.29134 | 
| 2 | RTX 2070s | [16L, 3000L] | float32 | 1 | 0.13398 |
| 3 | RTX 2070s | [16L, 10000L] | float16 | 5 | 0.1502 |
| 4 | RTX 2070s | [16L, 3000L] | float16 | 1 | 0.06901 |


API文档 https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/kthvalue_cn.html

## 1.2 业内方案调研

Pytorch中对kthvalue算子基于GPU计算实现,  forward整体性能如下(基于pytorch　v1.12)：

| Case No. | device | input_shape | input_type | k | Pytorch Perf(ms) |
|---|---|---|---|---|---|
| 1 | RTX 2070s | [16L, 10000L] | float32 | 5 | 0.08037 | 
| 2 | RTX 2070s | [16L, 3000L] | float32 | 1 | 0.041758 |
| 3 | RTX 2070s | [16L, 10000L] | float16 | 5 | 0.070236 |
| 4 | RTX 2070s | [16L, 3000L] | float16 | 1 | 0.027326 |

## 1.3 对比分析

目前Paddle与Pytorch的API设计方案相似，两种case下测试Pytorch性能更优,
二者主要差别是Paddle采用的是cub方式计算，🕑然而Pytorch采用的是基数排序RadixSelect方式大大提升了性能。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

可参考topk算子, 通过使用飞桨内部已经实现的RadixSearch，优化现在的cub方式的排序计算，预期性能提升2.7倍以上。

## 2.2 Host端计算流程

将输入转置到最后一维度，优化配置相应的grid和block。

## 2.4 Device端计算流程

对kthvalue, 用kernel嵌套的方式，对每一个目标维度上基于基数排序方式计算kth-value。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2023-02-22 |
| 2 | 完成开发文档设计  | 2023-03-12 |
| 3 | kthvalue优化实现  | 2023-03-31 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2023-04-15 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)


