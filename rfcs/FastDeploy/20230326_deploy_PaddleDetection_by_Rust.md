# 使用Rust完成PaddleDetection模型部署设计文档

| 领域                                                       | 飞桨文档体验方案                                       | 
|----------------------------------------------------------|------------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | wanziyu                                        | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-26                                     | 
| 版本号                                                      | V0.0                                           |  
| 文件名                                                      | 20230326_deploy_PaddleDetection_by_Rust.md<br> | 

# 一、概述

## 1、相关背景

随着飞浆项目使用者越来越多，需要开发Rust部署示例来满足Rust开发者的模型部署需求。

## 2、功能目标
* 在`FastDeploy`中使用`Rust`完成`PaddleDetection`中`PP-YOLOE`, `PaddleYOLOv8`, `PaddleYOLOv5`等模型的部署。
* 完成中英文部署文档。

## 3、意义
使得Rust开发者也可以使用FastDeploy部署`PP-YOLOE`, `PaddleYOLOv8`, `PaddleYOLOv5`等目标检测模型。

# 二、飞桨现状
`FastDeploy`目前缺乏使用Rust部署目标检测模型的例子。

# 三、设计思路与实现方案

## 总体思路
在Rust层面进行目标监测模型部署可以通过Rust的`bindgen`工具生成与FastDeploy C API和动态库的绑定。

##定义模型部署方法
在Rust层面调用C API中的目标检测模型对应的方法，如`Predict`和`Initialize`等。 同时需要符合Rust使用者的编程习惯。

# 四、测试和验收的考量
在Rust层面实现部署`YOLOv5`、`YOLOv8`和`PP-YOLOE`检测模型，并展示预测成功后的可视化图片。

# 五、可行性分析和排期规划
* 通过Rust和`bindgen`库调用FastDeploy预编译库中的模型部署C API和动态库`libfastdeploy.so`，进行模型部署实现代码开发, 方案可行。
* 预计需要一周进行开发。

# 六、影响面
Rust部署实现代码在FastDeploy项目的`examples/application/rust`目录下，对其他模块没有影响。
