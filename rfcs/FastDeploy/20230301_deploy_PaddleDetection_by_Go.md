# 使用Go完成PaddleDetection部署设计文档

| 领域                                                       | 飞桨文档体验方案                                     | 
|----------------------------------------------------------|----------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | wanziyu                                      | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-01                                   | 
| 版本号                                                      | V0.0                                         |  
| 文件名                                                      | 20230301_deploy_PaddleDetection_by_Go.md<br> | 

# 一、概述

## 1、相关背景

随着飞浆项目使用者越来越多，需要开发Golang部署示例来满足Golang开发者的模型部署需求。

## 2、功能目标
* 在`FastDeploy`中使用`Golang`完成`PaddleDetection`中`PP-YOLOE`, `PaddleYOLOv8`, `PaddleYOLOv5`等模型的部署。
* 完成中英文部署文档。

## 3、意义
使得Go开发者也可以使用FastDeploy部署`PP-YOLOE`, `PaddleYOLOv8`, `PaddleYOLOv5`等目标检测模型。

# 二、飞桨现状
`FastDeploy`目前缺乏使用Golang部署目标检测模型的例子。

# 三、设计思路与实现方案

## 总体思路
在Golang层面进行目标监测模型部署可以通过Golang调用`fastdeploy/vision`下的C++ API进行实现。

##定义模型结构体
在Golang层面定义`YOLOv5`、`YOLOv8`和`PP-YOLOE`等模型结构体，在结构体中调用C++ API中的各个模型类，并增加辅助字段等。

##定义模型部署方法
在Golang层面定义目标检测模型结构体对应的方法，如`Predict`和`Initialize`等，通过调用C++ API中的模型方法进行复用。 同时需要符合Golang使用者的编程习惯。

# 四、测试和验收的考量
在Golang层面实现部署`YOLOv5`、`YOLOv8`和`PP-YOLOE`检测模型，并进行测试。

# 五、可行性分析和排期规划
* 通过Golang调用`fastdeploy/vision`目录下的模型部署`C++ API`, 方案可行。
* 预计需要两周进行开发，一周进行部署测试。

# 六、影响面
Golang实现代码在FastDeploy项目的`examples/application/go`目录下，对其他模块没有影响。
