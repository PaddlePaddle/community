# 集成SOLOv2模型到FastDpeloy，并在Paddle Infenence、ONNX Runtime、TernsorRT后端测试验证

| 领域                                                       | 飞桨文档体验方案                                  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | Zheng-Bicheng                             | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-02-21                                | 
| 版本号                                                      | V0.0                                      | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==0.0                     | 
| 文件名                                                      | 20230221_add_solov2_for_fastdeploy.md<br> | 


# 一、概述

## 1、相关背景

完成SOLOv2算法前后处理，及精度对齐，开发Python部署示例和C++部署示例

## 2、功能目标

* Python和C++实现代码 ，在FastDeploy repo 的examples/vision/detection/paddledetection/jetson；
* 中英文文档，在FastDeploy repo 的examples/vision/detection/paddledetection/jetson；包含从零的环境安装文档
* 提交benchmark测试数据及精度对齐数据

## 3、意义

补齐FastDeploy在PaddleDetection上无法部署SOLOv2的遗憾。

# 二、设计思路与实现方案

## 删除ApplyDecodeAndNMS改为ApplyNMS

不带decode的模型几乎在部署中见不到，后续RK指定特殊节点进行导出的模型可以移动至RKYOLO仓库。
因此可以考虑删除ppdet_decode，PPDet模型只保留以下三种情况：

- 仅删除NMS的模型
- 带所有后处理的模型
- Solov2



## 删除ProcessUnDecodeResults并新增三个不同的Process

删除ProcessUnDecodeResults，新增以下三个api:

- ProcessGeneral（仅删除NMS的模型,带所有后处理的模型）
- ProcessSolov2(Solov2)

其中ProcessGeneral需要添加一个参数bool with_nms = false，用于区别删除NMS的模型与其他模型。

## PreProcess读取yaml文件的代码移动到PPDet模型下，并赋值给PreProcess和PostProcess

如果postprocess根据arch来判断后处理执行哪个代码需要读取yaml文件，使用这种方法可以节省一次读取yaml文件的时间

# 三、测试和验收的考量

* 验收标准：先提交精度与速度报告，待报告通过后，提交pr到FastDeploy仓库。

# 四、可行性分析和排期规划

对各主流深度学习框架已经有一定了解，需要进一步做细致的体验测试及分析。
预计整体的工作量在三周内可完成，不会晚于黑客松设定的验收 DDL。


# 五、影响面

删除了ppdet_decode,部分代码会收到影响，例如没有decode版本的picodet会无法部署。
不过考虑到几乎没有用户使用这个版本的decode，删除ppdet_decode能让代码不会非常耦合，是值得的。
