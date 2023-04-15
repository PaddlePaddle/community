# 关于 FastDeploy 中 Paddle Lite GPU（OpenCL） 适配的征求意见稿

| 任务名称                                                       | 完成FastDeploy中Paddle Lite GPU的适配，并完成批量测试脚本，完成10+模型的测试                                  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者   | UnseenMe  | 
| 提交时间   | 2023-04-08 | 
| 版本号  | V1.0 | 
| 依赖飞桨版本 | FastDeploy develop;  Paddle-Lite v2.13-rc              | 
| 文件名  | 20230408_ph4_244_add_paddlelitegpu_to_fastdeploy.md | 


# 一、概述

## 1、相关背景

 - FastDeploy 是一款全场景、易用灵活、极致高效的AI推理部署工具，支持云边端部署，支持160+模型开箱即用。FastDeploy 的后端已经支持了 Paddle Lite，但是 Paddle Lite GPU 还没有适配。
 - Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及边缘端在内的多种硬件平台。已经完成了对 OpenCL 的适配。

## 2、功能目标

* 完成 Paddle Lite GPU 后端接入 FastDeploy 工作。
* 在Paddle Lite 安卓后端实现的10+模型上测试，测试模型运行正确（至少完成OpenCL的适配 ）。
* 完成相关中英文文档。

## 3、意义

为 FastDeploy 进一步丰富后端对 Android Arm OpenCL 的支持能力。

# 二、实现方案

 1. 搭建 FastDeploy 开发环境  
 可以参考[CPU部署库编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)搭建。
 1. 修改各类枚举变量  
在enum_variables.h文件中增加对arm OpenCL的支持。
 1. 实现backend接口  
 因为LiteArmCPU已经在fastdeploy\runtime\backends\lite中实现，所以本次不用新增目录，在这里修改即可。
 1. Backend集成进FastDeploy Runtime  
 因为LiteArmCPU已经被支持，这里需要仔细检查一下参数。
 1. 编译CMake配置  
 同上因为Lite后端已经被支持，这里也需要仔细检查一下参数。
 1. C++后端测试  
写一个新的OpenCL后端示例，进行加载模型并推理测试。

# 三、测试和验收的考量

 - 在 Paddle Lite 安卓后端实现的10+模型上测试。
 - 测试时重点关注 benchmark 测试数据及精度对齐数据。

# 四、影响面

FastDeploy 的结构决定了，新增的后端支持对原内容影响很小。需要在已有文件中追加内容时（比如 [enum_variables.h](https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/runtime/enum_variables.h)）要多加注意，不要影响到其他后端支持。

# 五、排期规划

2023/4/08 提交 RFC  
2023/4/20 提交 精度与速度报告  
2023/4/25 提交 PR  
2023/4/30 合并 PR  

# 名词解释

 - OpenCL : Open Computing Language（开放运算语言）
