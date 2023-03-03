# 完成pp-ocrv3在RK3588上的部署，并验证正确性

| 领域                                                       | 飞桨文档体验方案                                  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | Zheng-Bicheng                             | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-21                                | 
| 版本号                                                      | V0.0                                      | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==0.0                     | 
| 文件名                                                      | 20230221_deploy_ppocrv3_for_rknpu2.md<br> | 


# 一、概述

## 1、相关背景

完成PP-OCRv3模型转换，并量化完成在在RK3588上的部署，确定精度和推理速度正确

## 2、功能目标

* 提交适配代码，及对应的中英文文档
* 提交PP-OCRv3的部署示例
* 提交benchmark测试数据及精度对齐数据

## 3、意义

完善FastDeploy在RKNPU2上的OCR部署。

# 二、设计思路与实现方案

### Det部分

- RKNPU2不支持Normalzie 和 Permute,参考PPYOLOE在FastDeploy中的部署,因此添加了DisableNormalize 和DisablePermute
- RKNPU2 不支持动态shape，因此参考Rec部分新增了固定shape推理。

### Cls部分

- RKNPU2不支持Normalzie 和 Permute,参考PPYOLOE在FastDeploy中的部署,因此添加了DisableNormalize 和DisablePermute

### Rec部分

- RKNPU2不支持Normalzie 和 Permute,参考PPYOLOE在FastDeploy中的部署,因此添加了DisableNormalize 和DisablePermute

### OCR Result部分

- 在显示result时会出现rec_score为0但是仍然被框出来的情况，这里对VisOcr函数新增了参数score_threshold


# 三、测试和验收的考量

* RKNPU2对PPOCR-V3 Rec部分量化支持度较差，因此可能只考虑量化Cls和Det两个部分。

# 四、可行性分析和排期规划

对各主流深度学习框架已经有一定了解，需要进一步做细致的体验测试及分析。
预计整体的工作量在三周内可完成，不会晚于黑客松设定的验收 DDL。


# 五、影响面

都是新增API，对现有的FastDeploy架构不造成影响
