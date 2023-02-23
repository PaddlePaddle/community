# Arm 虚拟硬件上完成 PP-OCR 文本检测模型的部署与优化设计文档

|任务名称 | Arm 虚拟硬件上完成 PP-OCR 文本检测模型的部署与优化 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | txyugood | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-2-22 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230222_ppocr_deploy_for_arm.md<br> | 

# 一、概述
## 1、相关背景
将 PaddleOCR 模型库中的文本检测模型 (Text Detection Model) 部署在 Arm Cortex-M55 处理器上并使用 Arm 虚拟硬件 Corstone-300 平台进行验证。
[https://github.com/PaddlePaddle/Paddle/issues/50632#task214](https://github.com/PaddlePaddle/Paddle/issues/50632#task214)
## 2、功能目标
将PPOCR中的文本检测模型通过TVM编译后，在ARM虚拟硬件上运行，并输出推理结果。
## 3、意义
为PPOCR套件提供边缘计算方案。

# 二、飞桨现状

目前PPOCR支持英文检测模型在ARM虚拟硬件上的运行。

# 三、业内方案调研
通过检索没有找到完整的pytorch与tensorflow的虚拟ARM部署方案。

# 四、对比分析

Paddle-examples-for-AVH中的例子可以在虚拟ARM上跑通全流程，但在本次调研中也发现了文档中一些不详细的地方，遇到一些问题，通过查阅资料解决，会在该项目中进行完善文档。


# 五、设计思路与实现方案

## 1、主体设计思路与折衷
参考实例代码跑通环境部署、TVM安装、模型量化、模型编译、应用程序编写与测试等部署。
### 主体设计具体描述
1.通过Docker镜像部署环境。
2.在镜像中安装TVM工具。
3.量化并编译模型。
4.编写基于CMSIS-NN的应用程序。
5.在本地FVP环境中运行测试。
6.在远程AMI环境运行测试。

### 主体设计选型考量
选择ch_ppocr_mobile_v2.0_det模型主要原因，
1.模型参数规模小，mobile模型适合运行在嵌入式设备上。
2.该模型未被量化，可在编译阶段先对模型进行适当的量化，尽可能多的调用CMSIS-NN库。


## 2、关键技术点/子模块设计与实现方案
模型量化与编译是本项目重点内容，可参考PPOCR中的模型量化与裁剪方法。

## 3、主要影响的模块接口变化
单独的demo项目不影响飞桨框架。

# 六、测试和验收的考量
通过本地FVP和远程AMI运行程序，可使用中文图片正常运行并检测结果正确。

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
* 量化与编译模型（2023-2-22至2023-2-28）
* 应用程序编写(2023-3-1至2023-3-3)
* FVP环境测试(2023-3-1至2023-3-3)
* AMI环境测试(2023-3-3至2023-3-5)
* 提交PR(2023-3-6)

# 名词解释
FVP:ARM的固定虚拟平台


# 附件及参考资料
