# 关于PP-Strucure接入FastDeploy的征求意见稿

| 任务名称 | PP-Strucure接入FastDeploy，并在Paddle Infenence、ONNX Runtime、TernsorRT、Openvino后端测试验证 | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者   | UnseenMe  | 
| 提交时间   | 2023-04-08 | 
| 版本号  | V1.0 | 
| 依赖飞桨版本 | FastDeploy develop | 
| 文件名  | 20230408_ph4_243_add_ppstrucure_to_fastdeploy.md | 


# 一、概述

## 1、相关背景

 - FastDeploy 是一款全场景、易用灵活、极致高效的AI推理部署工具，支持云边端部署，支持160+模型开箱即用。
 - PP-Structure 是一款智能文档分析系统，可以帮助开发者完成版面分析、表格识别等文档理解相关任务。其中 ser_VI-LayoutXLM_xfund_zh 模型提供强大的关键信息抽取(Key Information Extraction，KIE)能力。

## 2、功能目标

* 完成 ser_VI-LayoutXLM_xfund_zh 模型前后处理，及精度对齐。
* 完成开发Python部署示例和C++部署示例开发。
* 完成相关中英文文档。

## 3、意义

为 FastDeploy 增加 KIE 能力。

# 二、实现方案

 1. 搭建 FastDeploy 开发环境  
可以参考[CPU部署库编译](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)搭建。
 1. 准备模型  
可以从[模型库](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/models_list.md)取得模型。
 1. C++ 代码实现  
需要分别实现前处理，后处理与模型专属h文件cc文件。
 1. Python接口封装  
需要创建模型专属Python文件，Pybind文件与调用Pybind函数。
 1. 精度与速度测试  
C++与Python需要分别进行测试。
 1. 示例代码开发  
C++部分需要CMake文件，cc文件和README文件。  
Python部分需要infer.py与README文件。  
另外还需要整体介绍的README文件。

# 三、测试和验收的考量

 - 需要将 C++ 与 Python 实现分别测试。
 - 测试时重点关注精度与速度。

# 四、影响面

FastDeploy 的结构决定了，新增的部署示例对原内容影响很小。需要在已有文件中追加内容时（比如 [vision.h](https://github.com/PaddlePaddle/FastDeploy/blob/develop/fastdeploy/vision.h)）要多加注意，不要改动原内容。

# 五、排期规划

2023/4/08 提交 RFC  
2023/4/15 提交 精度与速度报告  
2023/4/20 提交 PR  
2023/4/30 合并 PR  

# 名词解释

 - KIE : Key Information Extraction（关键信息抽取）

# 附件及参考资料

 - LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich
   Document Understanding [2104.08836.pdf
   (arxiv.org)](https://arxiv.org/pdf/2104.08836.pdf)
 - [PP Sturcture KIE](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/kie/README_ch.md)
 - [FastDeploy/develop_a_new_model.md at develop · PaddlePaddle/FastDeploy · GitHub](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/develop_a_new_model.md)
