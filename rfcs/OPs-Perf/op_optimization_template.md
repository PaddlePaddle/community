# 标题

标题如：Transpose OP性能优化设计文档
| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 提交作者<input type="checkbox" class="rowselector hidden">   | xxx                                               |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-05-31                                                   |
| 版本号                                                       | 此设计文档的版本号，如V1.0                                   |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 无特殊情况，均基于PaddleDevelop版本开发                      |
| 文件名                                                       | 提交的markdown设计文档文件名称，如：20220531_xxx_op_optimization.md<br> |


# 1 背景与意义

填写此任务的开发背景，为什么想要优化这个OP。如果有相关issue，请将issue链接填写至此。

## 1.1 飞桨现状

对于此OP在目前飞桨框架（Develop分支）中的性能现状调研，表格形式列出[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)中各种case场景下的OP性能数据。


## 1.2 业内方案调研

调研业内深度学习框架中此OP的性能现状，范围包括不限于PyTorch、TensorFlow等，表格形式列出1.1中全部case下的OP性能数据。

## 1.3 对比分析

对比1.1和1.2中的表格，结合对比结果分析Paddle与业内最优实现方案的优劣，具体地，需要包含[OP Benchmark](https://github.com/PaddlePaddle/benchmark/tree/master/api/tests_v2)中各类case的性能优劣。

# 2 设计方案与性能预期

结合分析结论，设计符合Paddle规则的OP性能优化方案，并预估性能提升效果。

## 2.1 关键模块与性能提升点

详述新增模块的功能特点，以及关键性能提升点。

## 2.2 Host端计算流程

详述Host端的计算流程。

## 2.4 Device端计算流程

详述Host端和Device端的计算流程。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)

（需要附上OP性能优化效果对比表格）

# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone



# 5 影响面

需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
