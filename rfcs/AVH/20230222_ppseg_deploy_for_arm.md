# [Hackathon No.218] Arm 虚拟硬件上完成 PaddleSeg 模型的部署

| 任务名称                                                     | [Hackathon No.218] Arm 虚拟硬件上完成 PaddleSeg 模型的部署 | 
|----------------------------------------------------------|------------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | ZhengBicheng                                   | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-02                                     | 
| 版本号                                                      | V1.0                                           | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                        | 
| 文件名                                                      | 20230222_ppseg_deploy_for_arm.md<br>           | 

# 一、概述
## 1、相关背景

将 PaddleSeg 模型库中的目标分割模型 (Segmentation Model) 部署在 Arm Cortex-M55 处理器上并使用 Arm 虚拟硬件 Corstone-300 平台进行验证。

## 2、功能目标

使用Arm虚拟硬件完成目标分割应用的结果验证。

## 3、意义

为PaddleSeg套件提供边缘计算方案。

# 二、飞桨现状

目前PaddleSeg暂不支持目标分割模型在ARM虚拟硬件上的运行。

# 三、业内方案调研

以下两个方案来自[20230222_ppocr_det_deploy_for_arm](./20230222_ppocr_det_deploy_for_arm.md):

* [语音识别](https://arm-software.github.io/AVH/main/examples/html/MicroSpeech.html)
    
    该项目识别两个关键字Yes和No。使用Tensorflow Lite来实现识别模型，可以运行在ARM虚拟硬件上。


* [行人检测](https://github.com/apache/tvm/tree/main/apps/microtvm/cmsisnn)

    该项目使用TVM对TensorFlow Lite导出的模型文件进行编译，然后基于CMSIS-NN运行在Cortex(R)-M55 CPU上。

# 四、对比分析

以上方案均基于TensorFlow Lite实现，本方案将使用飞桨导出的静态图模型，使用TVM量化编译后再ARM虚拟硬件上运行，并输出具有可读性的运行结果。


# 五、设计思路与实现方案

## 1、主体设计思路与折衷

参考实例代码跑通TVM安装、模型编译、代码编写与测试等部署。

* TVM安装部分将提供TVM安装教程。
* 模型编译部分将采用tvmc来编译模型，最后生成能够在Arm Cortex-M55设备上运行的模型
* 代码编写采用c++或c来编写，可能考虑附带上一定的python脚本
* 测试时采用图片输入的形式，输出由0，1组成的矩阵以及数据文件，最后会传回ubuntu复现图片。

### 主体设计具体描述

将提供模型转换文档，模型部署代码，以及展示测试结果

### 主体设计选型考量

按照任务需求使用[PPHumanSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.7/contrib/PP-HumanSeg)部署

## 2、关键技术点/子模块设计与实现方案

按PPOCR demo实现类似的方案

## 3、主要影响的模块接口变化

无影响

# 六、测试和验收的考量

使用Arm虚拟硬件平台验证目标分割模型应用运行结果, 检测结果正常并具有可读性。

# 七、影响面

无影响与风险

# 八、排期规划

还是存在一定难度的，将会尽快完成代码。

* 转换模型(3-15前)
* 部署模型(4-15前)
* 输出demo(5-15前)
