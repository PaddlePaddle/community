# Arm 虚拟硬件上完成 PP-OCR 文本检测模型的部署与优化设计文档

|任务名称 | Arm 虚拟硬件上完成 PP-OCR 文本检测模型的部署与优化 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | txyugood | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-2-22 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230222_ppocr_det_deploy_for_arm.md<br> | 

# 一、概述
## 1、相关背景
将 PaddleOCR 模型库中的文本检测模型 (Text Detection Model) 部署在 Arm Cortex-M55 处理器上并使用 Arm 虚拟硬件 Corstone-300 平台进行验证。
[https://github.com/PaddlePaddle/Paddle/issues/50632#task214](https://github.com/PaddlePaddle/Paddle/issues/50632#task214)
## 2、功能目标
使用Arm虚拟硬件完成文本检测应用的结果验证。
## 3、意义
为PPOCR套件提供边缘计算方案。

# 二、飞桨现状

目前PPOCR支持英文检测模型在ARM虚拟硬件上的运行。

# 三、业内方案调研

* [语音识别](https://arm-software.github.io/AVH/main/examples/html/MicroSpeech.html)
    
    该项目识别两个关键字Yes和No。使用Tensorflow Lite来实现识别模型，可以运行在ARM虚拟硬件上。


* [行人检测](https://github.com/apache/tvm/tree/main/apps/microtvm/cmsisnn)

    该项目使用TVM对TensorFlow Lite导出的模型文件进行编译，然后基于CMSIS-NN运行在Cortex(R)-M55 CPU上。

# 四、对比分析

以上方案均基于TensorFlow Lite实现，本方案将使用飞桨导出的静态图模型，使用TVM量化编译后再ARM虚拟硬件上运行，并输出具有可读性的运行结果。


# 五、设计思路与实现方案

## 1、主体设计思路与折衷
参考实例代码跑通环境部署、TVM安装、模型量化、模型编译、应用程序编写与测试等部署。
### 主体设计具体描述
1. 选择模型。

2. 量化模型

3. 编译模型(tvmc)

4. 应用程序编写(前后端处理)

5. 使用Arm虚拟硬件运行应用

6. 验证运行结果。

### 主体设计选型考量
选择ch_ppocr_mobile_v2.0_det模型主要原因，
1.模型参数规模小，mobile模型适合运行在嵌入式设备上。
2.该模型未被量化，可在编译阶段先对模型进行适当的量化，尽可能多的调用CMSIS-NN库。


## 2、关键技术点/子模块设计与实现方案
模型量化与编译是本项目重点内容，学习研究TVM中的量化方法。

## 3、主要影响的模块接口变化
单独的demo项目不影响飞桨框架。

# 六、测试和验收的考量
使用Arm虚拟硬件平台验证文本检测应用运行结果, 检测结果正常并具有可读性。

# 七、影响面

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

# 八、排期规划
* 环境搭建并跑通参考项目（已完成）
* 量化与编译模型（2023-2-28至2023-3-5）
* 应用程序编写(2023-3-5至2023-3-6)
* AVH测试(2023-3-6至2023-3-8)
* 提交PR(2023-3-9)

# 名词解释
AVH： Arm虚拟硬件（Arm Virtual Hardware)

# 附件及参考资料
