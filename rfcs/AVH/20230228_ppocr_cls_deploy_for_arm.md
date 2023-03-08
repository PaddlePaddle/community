# Arm 虚拟硬件上完成 PP-OCR 方向检测 模型的部署

| 任务名称                                                     | Arm 虚拟硬件上完成 PP-OCR 方向检测 模型的部署            | 
|----------------------------------------------------------|------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | ZhengBicheng                             | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-28                               | 
| 版本号                                                      | V0.0                                     | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                  | 
| 文件名                                                      | 20230228_ppocr_cls_deploy_for_arm.md<br> | 

# 一、概述

## 1、相关背景

任务目标为将 ch_ppocr_mobile_v2.0_cls 部署在 Arm Cortex-M55 处理器上并使用 Arm 虚拟硬件 Corstone-300 平台进行验证。

## 2、功能目标

使用Arm虚拟硬件完成文本方向分类模型应用的结果验证。

## 3、意义

为ch_ppocr_mobile_v2.0_cls套件提供TVM部署的实例方案。

# 二、飞桨现状

ch_ppocr_mobile_v2.0_cls暂时没有在TVM上运行的案例。

# 三、业内方案调研

* [PPOCR部署](https://github.com/ArmDeveloperEcosystem/Paddle-examples-for-AVH/tree/main/OCR-example)中提供了PPOCR模型在TVM上运行的实例代码。

# 四、对比分析

本方案将采用类似的方案，使ch_ppocr_mobile_v2.0_cls在ARM硬件上部署，并输出具有可读性的运行结果。


# 五、设计思路与实现方案

## 1、主体设计思路与折衷

参考实例代码跑通TVM安装、模型编译、代码编写与测试等部署。

* TVM安装部分将提供TVM安装教程。
* 模型编译部分将采用tvmc来编译模型，最后生成能够在Arm Cortex-M55设备上运行的模型
* 代码编写采用c++或c来编写，可能考虑附带上一定的python脚本
* 测试时采用图片输入的形式，输出类别与置信度

### 主体设计具体描述

将提供模型转换文档，模型部署代码，以及展示测试结果

### 主体设计选型考量

按照任务需求使用ch_ppocr_mobile_v2.0_cls部署


## 2、关键技术点/子模块设计与实现方案

按PPOCR demo实现类似的方案

## 3、主要影响的模块接口变化

无影响

# 六、测试和验收的考量

使用Arm虚拟硬件平台验证文本方向分类模型应用运行结果, 检测结果正常并具有可读性。

# 七、影响面

无影响与风险

# 八、排期规划

还是存在一定难度的，将会尽快完成代码。

* 转换模型(3-15前)
* 部署模型(4-15前)
* 输出demo(5-15前)
