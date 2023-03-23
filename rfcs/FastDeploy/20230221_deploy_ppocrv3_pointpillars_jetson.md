# 完成PointPillars集成到FastDeploy，并在Jetson Orin硬件上部署验证精度和速度。

| 领域                                                       | 飞桨文档体验方案                                  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | 无名                            | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-23                                | 
| 版本号                                                      | V1.0                                      | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==2.4.2                     | 
| 文件名                                                      | 20230221_deploy_ppocrv3_pointpillars_jetson.md<br> | 


# 一、概述

## 1、相关背景

将PointPillars集成到FastDeploy，并在Jetson Orin硬件上部署验证精度和速度。
https://github.com/PaddlePaddle/Paddle3D


## 2、功能目标

* 提交适配代码，及对应的中英文文档
* 提交PointPillars的部署示例
* 提交benchmark测试数据及精度对齐数据

## 3、意义

完善FastDeploy在 Jetson Orin 上的Paddle3D部署。

# 二、设计思路与实现方案

### Det部分
- 前处理类实现，创建 Preprocess 类
- 声明Run、preprocess、LetterBox和构造函数，以及必要的变量及其set和get方法
- 后处理类实现，创建 Postprocess 类
- 声明Run、preprocess、LetterBox和构造函数，以及必要的变量及其set和get方法

### Cls部分

- 前处理类实现，创建 Preprocess 类
- 声明Run、preprocess、LetterBox和构造函数，以及必要的变量及其set和get方法
- 后处理类实现，创建 Postprocess 类
- 声明Run、preprocess、LetterBox和构造函数，以及必要的变量及其set和get方法

### Rec部分

- 前处理类实现，创建 Preprocess 类
- 声明Run、preprocess、LetterBox和构造函数，以及必要的变量及其set和get方法
- 后处理类实现，创建 Postprocess 类
- 声明Run、preprocess、LetterBox和构造函数，以及必要的变量及其set和get方法

###  Result部分

- 前处理类实现，创建 Preprocess 类
- 声明Run、preprocess、LetterBox和构造函数，以及必要的变量及其set和get方法
- 后处理类实现，创建 Postprocess 类
- 声明Run、preprocess、LetterBox和构造函数，以及必要的变量及其set和get方法

# 三、测试和验收的考量

*  在Jetson Orin 运行 PointPillars模型，提交benchmark测试数据及精度对齐数据。

# 四、可行性分析和排期规划

对各主流深度学习框架已经有一定了解，需要进一步做细致的体验测试及分析。
预计整体的工作量在三周内可完成，不会晚于黑客松设定的验收 DDL。


# 五、影响面

都是新增API，对现有的FastDeploy架构不造成影响
