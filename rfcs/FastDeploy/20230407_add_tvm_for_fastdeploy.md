# 完成TVM接入FastDeploy，并在PP-YOLOE模型上验证正确性

| 领域                                                       | 飞桨文档体验方案                               | 
|----------------------------------------------------------|----------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | Zheng-Bicheng                          | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-21                             | 
| 版本号                                                      | V0.0                                   | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==0.0                  | 
| 文件名                                                      | 20230407_add_tvm_for_fastdeploy.md<br> | 


# 一、概述

## 1、相关背景

详细描述:
需要完成TVM接入Fastdeploy工作，并在CPU/GPU硬件上完成PP-YOLOE基于TVM后端的测试，确定精度和推理速度正确。

## 2、功能目标

提交内容：
pr：提交适配代码，及对应的中英文文档
pr：提交PP-YOLOE的部署示例
邮件：提交benchmark测试数据及精度对齐数据。

## 3、意义

为FastDeploy新增TVM后端

# 二、设计思路与实现方案

* 参考RKNPU的移植经验，设计继承自`BaseBackend`的`TVMBackend`类，同时实现其必备的几个虚函数
* 如果TVM有其独特的函数，考虑设计成static类型供外部调用
* 使用tvm转换PPYOLOE模型，并加载调用
* 测试模型精度

# 三、测试和验收的考量

* 验收标准：先提交精度与速度报告，待报告通过后，提交pr到FastDeploy仓库。

# 四、可行性分析和排期规划

对各主流深度学习框架已经有一定了解，需要进一步做细致的体验测试及分析。
预计整体的工作量在三周内可完成，不会晚于黑客松设定的验收 DDL。


# 五、影响面

对FastDeploy其他部分不造成影响
