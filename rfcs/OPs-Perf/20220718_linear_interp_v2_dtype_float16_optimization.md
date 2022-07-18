# IndexSample OP性能优化设计文档


| 基本信息                                                     | 内容                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------- |
| 提交作者<input type="checkbox" class="rowselector hidden">   | justld   |                                         
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2022-07-18 |                                                
| 版本号                                                 | V1.0  |                       
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden">| PaddleDevelop|                      
| 文件名                    | 20220718_linear_interp_v2_dtype_float16_optimization.md<br> |


# 1 背景与意义
目前，Paddle中的linear_interp_v2算子不支持float16类型，训练过程中无法使用混合精度训练来节约显存，提升训练速度。

## 1.1 飞桨现状

Paddle中的linear_interp_v2算子不支持float16类型，需要添加float16类型支持。

## 1.2 业内方案调研

Paddle中的linear_interp_v2算子目前仅支持float32、float64和uint8，不支持float16，混合精度训练时无法节约显存和加快训练速度。

## 1.3 对比分析

由于Paddle中的linear_interp_v2算子目前不支持float16，因此需要添加float16类型支持，具体方案如下：

```
PD_REGISTER_KERNEL(bilinear_interp_v2,
                   GPU,
                   ALL_LAYOUT,
                   phi::BilinearInterpKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int) {}
```

# 2 设计方案与性能预期

## 2.1 关键模块与性能提升点

根据 [paddle.nn.functional.interpolate](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/interpolate_cn.html#interpolate) 的api参数定义，输入参数`x`类型为float32、float64或uint8。那么优化后的类型额外支持float16类型，降低数据占用的显存，预计op显存占用降低一半。

## 2.2 Host端计算流程

无需修改已有api代码逻辑。

## 2.4 Device端计算流程

无需修改已有api代码逻辑。

# 3 测试和验收的考量

测试考虑的case如下：

- 编程范式场景：覆盖动态图测试场景
- 硬件场景：覆盖GPU测试场景
- Tensor精度场景：支持float16， float32 ， float64， uint8
- 参数组合场景
- 计算精度：float16和float32前向计算误差不超过1e-3
- 异常测试：由于在已有的API添加了类型支持float16, 所以需要做数据类型的异常测试
- 性能测试：使用float16计算性能不差于float32


验收考虑的case如下：

- Tensor精度场景：支持float16， float32 ， float64， uint8
- 计算精度：float16和float32前向计算误差不超过1e-3
- 性能测试：使用float16计算性能不差于float32


# 4 可行性分析和排期规划

本方案仅需在已有的API上添加float16类型支持。


# 5 影响面

在原有的API上添加新的类型支持，对其他模块无影响。



# 名词解释

无

# 附件及参考资料
无

