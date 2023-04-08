# 将PP-YOLOE-R在算能BM1684部署。利用FastDeploy，将PP-YOLOE-R在算能BM1684X部署

| 领域                                                       | 飞桨文档体验方案                               | 
|----------------------------------------------------------|----------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | thunder95                          | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-04-08                             | 
| 版本号                                                      | V0.0                                   | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==0.0                  | 
| 文件名                                                      | 20230408_add_bm1684x_ppyoloer_for_fastdeploy.md<br> | 


# 一、概述

## 1、相关背景

详细描述:
完成PP-YOLOE-R，在Fastdeploy中完成算法前后处理,开发Python部署示例和C++部署示例，确定精度和推理速度正确。

## 2、功能目标

提交内容：
pr：提交适配代码，及对应的中英文文档
pr：提交PP-YOLOE-R的部署示例
邮件：提交benchmark测试数据及精度对齐数据。

## 3、意义

为FastDeploy新增PP-YOLOE-R, 并在算能硬件上实现部署。

# 二、设计思路与实现方案

* FastDeploy repo 的examples/vision/detection/paddledetection/sophgo示例，完成ppyoloe-r相关功能开发，并在cpu/gpu验证
* 在算能的docker开发环境，将paddle模型转换bmodel
* 在算能设备上编译fastdeploy并跑通模型推理
* 测试模型精度和推理速度

# 三、测试和验收的考量

* 验收标准：先提交精度与速度报告，待报告通过后，提交pr到FastDeploy仓库。

# 四、可行性分析和排期规划

对各主流深度学习框架已经有一定了解，需要进一步做细致的体验测试及分析。
预计整体的工作量在三周内可完成，不会晚于黑客松设定的验收 DDL。


# 五、影响面

对FastDeploy其他部分不造成影响

