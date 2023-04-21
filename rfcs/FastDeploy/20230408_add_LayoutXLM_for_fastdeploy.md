# 集成PP-Strucure中ser_VI-LayoutXLM_xfund_zh模型接入FastDeploy，并在Paddle Infenence、ONNX Runtime、TernsorRT、Openvino后端测试验证 

| 领域                                                       | 飞桨文档体验方案                                  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                             | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-04-08                                | 
| 版本号                                                      | V0.0                                      | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==0.0                     | 
| 文件名                                                      | 20230408_add_LayoutXLM_for_fastdeploy.md<br> | 


# 一、概述

## 1、相关背景

完成ser_VI-LayoutXLM_xfund_zh算法前后处理，及精度对齐，开发Python部署示例和C++部署示例

## 2、功能目标

* Python和C++实现代码 ，在FastDeploy repo 的/examples/vision/ocr/PP-OCR；
* 中英文文档，在FastDeploy repo 的examples/vision/ocr/PP-OCR；包含从零的环境安装文档；包含从零的环境安装文档
* 提交benchmark测试数据及精度对齐数据

## 3、意义

补齐FastDeploy在PPOCR上无法部署表格识别模型SLANet的遗憾。

# 二、设计思路与实现方案

在fastdeploy/vision/ocr/ppocr下实现表格识别模型的前处理和后处理:
- ser_layoutxlm.h
- ser_layoutxlm_preprocessor.h
- ser_layoutxlm_postprocessor.h
模型初始化除了模型参数还需增加ser_dict_path参数

最终完成Paddle Infenence、ONNX Runtime、TernsorRT、Openvino四种后端的验证。

# 三、测试和验收的考量

* 验收标准：先提交精度与速度报告，待报告通过后，提交pr到FastDeploy仓库。

# 四、可行性分析和排期规划

对各主流深度学习框架已经有一定了解，需要进一步做细致的体验测试及分析。
预计整体的工作量在三周内可完成，不会晚于黑客松设定的验收 DDL。


# 五、影响面
单独新增的模型以及对应API， 对之前的几乎不产生影响。

