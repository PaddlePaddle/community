# Lerp OP性能优化设计文档


| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | WintersMontagne10335   |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-04-16 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20230416_lerp_op_optimization.md<br> |


# 1 背景与意义
目前，Paddle中Lerp OP采用第三方库组合实现，性能不足，有较大的提升空间。

## 1.1 飞桨现状

通过OP Benchmark测试Lerp OP在当前飞桨框架（Develop分支）的性能。测试数据如下表：
| Case No. | device |input_type | x_shape | y_shape | origin Paddle Perf(ms) |
|---|---|---|---|---|---|
| 1 | GeForce GTX960 | float32 | [-1L, 102400L] | [-1L, 102400L] | 0.6911145 | 
| 2 | GeForce GTX960 | float32 | [16L, 1L, 1L, 1L] | [16L, 3L, 224L, 224L] | 3.1153775 |
| 3 | GeForce GTX960 | float16 | [-1L, 102400L] | [-1L, 102400L] | 0.5005047 |
| 4 | GeForce GTX960 | float16 | [16L, 1L, 1L, 1L] | [16L, 3L, 224L, 224L] | 2.8568278 |

API文档：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/lerp_cn.html

## 1.2 业内方案调研

Pytorch中对应`paddle.lerp` 的API为 `torch.lerp`，对Lerp算子的实现基于GPU计算。整体性能如下：
| Case No. | device |input_type | x_shape | y_shape | PyTorch Perf(ms) | diff with origin Paddle |
|---|---|---|---|---|---|---|
| 1 | GeForce GTX960 | float32 | [-1L, 102400L] | [-1L, 102400L] | 0.5673498 | faster than 21.8% | 
| 2 | GeForce GTX960 | float32 | [16L, 1L, 1L, 1L] | [16L, 3L, 224L, 224L] | 0.8369803 | faster than 272.2% |
| 3 | GeForce GTX960 | float16 | [-1L, 102400L] | [-1L, 102400L] | 0.3154792 | faster than 58.6% |
| 4 | GeForce GTX960 | float16 | [16L, 1L, 1L, 1L] | [16L, 3L, 224L, 224L] | 0.6265184 |faster than 356.0%|

Pytorch中Lerp OP源码：https://github.com/pytorch/pytorch/blob/43e71cddb0dc85b43a98238740bd5f8584d841fd/aten/src/ATen/native/cuda/Lerp.cu

## 1.3 对比分析

- 整体性能Pytorch优于飞桨
- Case 2、Case 4都是有Broadcast的Case，此时Pytorch的速度大幅领先飞桨。这是因为飞桨的Lerp OP是用Eigen实现的。Eigen中的Broadcast过于复杂全面，严重拖慢了飞桨Lerp OP的速度。
- 除此之外，Pytorch对Weight为float且大于0.5、小于1.0的情况的进行了特殊处理。代码如下：

```C++
return (std::abs(weight_val) < 0.5)
    ? self_val + weight_val * (end_val - self_val)
    : end_val -
        (end_val - self_val) * (static_cast<T>(1) - weight_val);
```

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

- 优化读取、写入以及线程配置方法，预计整体性能将提升20%以上
- 使用飞桨充分优化的Elementwise Kernel，预计受Broadcast影响而性能较差的Case（如Case 2、Case 4）的速度将提升4倍以上
- 借鉴Pytorch对Weight为float且大于0.5、小于1.0的特殊情况的处理，预计这种情况下的性能将提升5%以上

## 2.2 Host端计算流程

- 向量化读取、写入
- 利用飞桨gpu_launch_config.h中的方法获得优良的线程配置
- 特殊处理Weight为float且大于0.5、小于1.0的情况

## 2.3 Device端计算流程

组合使用相关的Elementwise Kernel。

# 3 测试和验收的考量

参考：[算子性能优化验收标准](http://agroup.baidu.com/paddle-perf/md/article/4892913)

# 4 可行性分析和排期规划

时间和开发排期规划，主要milestone

| No. | 开发内容 | 预期时间 |
|---|---|---|
| 1 | 理清Paddle中OP设计思路，同类产品中最佳设计方案  | 2023-04-15 |
| 2 | 完成开发文档设计  | 2023-04-16 |
| 3 | 完成代码开发工作，并通过线程CI测试 | 2023-05-15 |


# 5 影响面

待优化的算子独立运行，不涉及其他算子和模块的修改，API设计与之前保持一致。

# 名词解释



# 附件及参考资料
[1]. [OP Benchmark使用指南](https://github.com/PaddlePaddle/benchmark/blob/master/api/README.md)
