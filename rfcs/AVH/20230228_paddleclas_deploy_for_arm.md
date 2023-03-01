# Arm虚拟硬件上完成PaddleClas模型的部署与优化文档

|任务名称 | Arm 虚拟硬件上完成 PaddleClas 模型的部署与优化 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | Boomerl | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-3-1 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230228_paddleclas_deploy_for_arm.md<br> | 

# 一、概述

## 1、相关背景
模型的训练和部署是AI的核心。百度和Arm达成合作，针对Arm Cortex-M处理器的深度学习应用场景，完成了PaddlePaddle模型在Cortex-M硬件上的适配。本任务将会从PaddleClas模型库中选取合适的模型，将其部署在Arm Cortex-M55（No.216）和Arm Cortex-M85处理器（No.217）上，并使用Arm虚拟硬件Corstone-300（No.216）和Corstone-310（No.217）平台完成验证。
[https://github.com/PaddlePaddle/Paddle/issues/50632#task217](https://github.com/PaddlePaddle/Paddle/issues/50632#task217)


## 2、功能目标
导出PaddleClas中的模型，使用TVM对PaddlePaddle模型进行编译，将模型部署在Arm虚拟硬件上，输出测试结果。

## 3、意义
为PaddleClas套件提供边缘计算方案。


# 二、飞桨现状
目前飞桨支持PaddlePaddle模型在Arm虚拟硬件上运行，同时配套有PaddleOCR和PaddleDetection模型在Arm虚拟硬件上的部署实践教程。


# 三、业内方案调研
通过检索发现tensorflow和pytorch都有部署到Arm硬件上的案例，tensorflow有自己的工具链tensorflow lite (micro)，pytorch则借助ONNX转换模型再通过诸如TVM等深度学习推理框架进行部署。
- [语音识别](https://arm-software.github.io/AVH/main/examples/html/MicroSpeech.html)
- [行人检测](https://github.com/apache/tvm/tree/main/apps/microtvm/cmsisnn)


# 四、对比分析
飞桨官网上的AVH部署实践教程只包括PaddleOCR和PaddleDetection的例子，没有包含PaddleClas，同时案例中没有详细说明模型是否量化，还有待完善。


# 五、设计思路与实现方案

## 1、主体设计思路
参考飞桨官网AVH部署教程完成环境部署、TVM安装、模型压缩（No.217要求）、模型编译、应用程序编写与测试等工作。

### 主体设计具体描述
1. 加载PaddleClas中的预训练模型并微调
2. 安装TVM和相关依赖，并完成环境测试
3. 模型压缩（No.217要求）
4. TVM编译模型
5. 编写应用程序
6. AVH实例中部署和测试
7. 验证运行结果
### 主体设计选型
1. No.216选择mobilenet_v2模型，
   - mobilenet_v2是移动视觉领域的经典模型，参数量小。
   - 推理框架对mobilenet_v2的算子优化好，便于部署。
2. No.217选择PPLCNet模型，
   - PPLCNet是百度针对Intel CPU设计的轻量化模型，通过部署在Arm芯片上测试其在不同架构芯片上的性能差异。

## 2、关键模块设计与实现方案
模型压缩（No.217要求）是本任务的重点，可以采取PTQ或QAT两种方案得到量化模型，再通过TVM编译部署到Arm虚拟硬件上。


# 六、测试与验收的考量
通过AVH部署并运行应用程序，输入为一张彩色图片，输出为分类结果以及推理时延等。


# 七、可行性分析和排期规划
任务No.216和No.217同步进行，No.216相比No.217少了量化压缩的步骤。
   - 环境搭建并跑通教程案例（2023-3-1至2023-3-3）
   - 模型训练与量化（2023-3-4至2023-3-20）
   - TVM编译模型（2023-3-21至2023-3-23）
   - 应用程序编写（2023-3-24至2023-4-10）
   - AVH部署和测试（2023-4-11至2023-4-13）
   - 提交PR（2023-4-14）


# 八、影响面

## 对用户的影响
单独Demo项目，对用户无影响。
## 对二次开发用户的影响
单独Demo项目，对用户无影响。
## 对框架架构的影响
无
## 对性能的影响
无
## 对比业内深度学习框架的差距与优势的影响
无
## 其他风险
无


# 名词解释
AVH：Arm Virtual Hardware，Arm虚拟硬件
PTQ：训练后量化
QAT：量化感知训练

# 附件及参考资料
[AVH动手实践(二) | 在Arm虚拟硬件上部署PP-OCR模型](https://www.paddlepaddle.org.cn/support/news?action=detail&id=3062)

[AVH动手实践(三) | 在Arm虚拟硬件上部署PP-PicoDet模型](https://www.paddlepaddle.org.cn/support/news?action=detail&id=3114)
