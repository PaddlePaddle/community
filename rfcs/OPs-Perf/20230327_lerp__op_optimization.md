# Prelu OP性能优化设计文档


| 基本信息                                                     | 内容                                   |
| ------------------------------------------------------------ |--------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | TanWei                            |
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-27                           |
| 版本号                                                       | V1.0                                 |
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | PaddleDevelop                        |
| 文件名                                                       | 20230327_lerp__op_optimization.md<br> |


# 1 背景与意义

  - 现状：目前 Paddle 内 lerp 算子采用第三方库组合实现，性能不足；
  - 目标：请优化计算实现，为 Paddle 优化 lerp op 在 GPU 上的计算性能，性能至少提升20%，针对性能差的case，性能至少提升4+倍。

## 1.1 飞桨现状

当前性能如下表(基于PaddlePaddle　develop分支)：
| Case No. | device | input_x_shape | input_x_type | input_y_shape | input_y_type | weight_type | Paddle Perf(ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | V100 | [16L, 102400L] | float32 |  [16L, 102400L]|float32| float32 | 0.05851 |
| 2 | V100 | [16L, 102400L] | float16 |  [16L, 102400L]|float16| float32 | 0.05324 |
| 3 | V100 | [16L, 1L, 1L, 1L]  | float32 |  [16L, 3L, 224L, 224L] |float32| float32 | 0.1695 |
| 4 | V100 | [16L, 1L, 1L, 1L]  | float16 |  [16L, 3L, 224L, 224L] |float16| float32 | 0.1653 |

飞桨当前版本没有实现float16版本的lerp算子，使用PD_REGISTER_KERNEL简单注册了float16用于测试。

API文档 https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/lerp_cn.html

## 1.2 业内方案调研

Pytorch中lerp算子实现了GPU版本，且实现了float16版本，forward整体性能如下(基于pytorch　v1.13.1)：

| Case No. | device | input_x_shape | input_x_type | input_y_shape | input_y_type | weight_type | Pytorch Perf(ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | V100 | [16L, 102400L] | float32 |  [16L, 102400L]|float32| float32 | 0.04246 |
| 2 | V100 | [16L, 102400L] | float16 |  [16L, 102400L]|float16| float32 | 0.03229 |
| 3 | V100 | [16L, 1L, 1L, 1L]  | float32 |  [16L, 3L, 224L, 224L] |float32| float32 | 0.04432 |
| 4 | V100 | [16L, 1L, 1L, 1L]  | float16 |  [16L, 3L, 224L, 224L] |float16| float32 | 0.04263 |
 
## 1.3 对比分析

目前Paddle与Pytorch的API设计方案相同。

Paddle中lerp的GPU实现使用了许多Eigen库的broadcast（速度比较慢，慢的原因在这个 PR ([#6229](https://github.com/PaddlePaddle/Paddle/pull/6229)) 中有描述）；Paddle中lerp没有float16的实现。

当input_x_shape和input_y_shape都为[16L, 102400L]，输入类型都为float32时，性能略低于pytorch。

当input_x_shape和input_y_shape都为[16L, 102400L]，输入类型都为float16时，性能低于pytorch。

当input_x_shape为[16L, 1L, 1L, 1L]，input_y_shape为[16L, 3L, 224L, 224L]时，paddle性能明显不足。

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

1、参考elementwise_add_kernel.cu实现，使用DEFINE_CUDA_ELEMENTWISE_OP，消除Eigen库的broadcast的依赖。
ps：查看DEFINE_CUDA_ELEMENTWISE_OP代码，其中已经有了线程配置和向量化读取等逻辑，不需要再加。
2、特化float16的lerp实现。

## 2.2 Host端计算流程

特化float16的lerp实现，float16在调用kernal前转换为float32类型，结束后转换回float16。

实现LerpFunctor，通过funcs::BroadcastKernel调用LerpFunctor。

## 2.4 Device端计算流程

根据lerp公式$lerp(x,y,weight)=x+weight∗(y−x)$实现LerpFunctor。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)



# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2023-03-27 |
| 2 | 完成开发文档设计  | 2023-03-28 |
| 3 | prelu优化实现  | 2023-03-31 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2023-04-10 |



# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。


# 名词解释


# 附件及参考资料

[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
[2]. [开发C++算子](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/new_cpp_op_cn.html)


