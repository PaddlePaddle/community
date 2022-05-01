# Paddle-Lite-Demo YoloV5设计文档

|名称 | Paddle-Lite-Demo YoloV5| 
|---|---|
|提交作者 | thunder95 | 
|提交时间 | 2022-03-29 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | paddlepaddle-gpu==2.2 | 
|文件名 | 20220329_yolov5_andriod.md | 


# 一、概述
## 1、相关背景

参考ssd_mobilnetv1目标检测的Android demo，使用 yolo_v5 模型在安卓手机上完成demo开发.

## 2、功能目标

输入为摄像头实时视频流，输出为包含检测框的视频流；在界面上添加一个backend选择开关，用户可以选择将模型运行在 CPU 或 GPU 上，保证分别运行CPU和GPU结果均正确。

## 3、意义

使用飞桨部署工具Paddle-Lite完成YoloV5模型在手机端的部署。


# 二、飞桨现状

飞桨开源了一套边缘段部署框架[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite), 包括手机安卓端。

Paddle Lite 是一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架，定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台。


# 三、业内方案调研

国内外主流边缘端部署框架罗列如下：

- [NCNN](https://github.com/Tencent/ncnn)  ncnn 是腾讯优图实验室首个开源项目，是一个为手机端极致优化的高性能神经网络前向计算框架, 跨平台，主要支持 android，次要支持 ios / linux / windows。
- [MACE](https://github.com/XiaoMi/mace) Mobile AI Compute Engine (MACE) 是一个专为移动端异构计算平台优化的神经网络计算框架。
- [Tensorflow-Lite](https://tensorflow.google.cn/lite) TensorFlow Lite 提供了转换 TensorFlow 模型，并在移动端（mobile）、嵌入式（embeded）和物联网（IoT）设备上运行 TensorFlow 模型所需的所有工具。
- [Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) 由飞桨开源的一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架。



# 四、对比分析

PyTorch本身不提供手机端推理框架，但[Ylov5官方](https://github.com/ultralytics/yolov5)提供了tf-lite模型的导出功能，并进一步实现了手机端部署推理。

Github上也有开源开发者基于[NCNN](https://github.com/cmdbug/YOLOv5_NCNN)， [TNN](https://github.com/cmdbug/TNN_Demo), [JNI](https://github.com/caowei110/uni-yolov5-android-jni)等移动端推理框架实现了YoloV5在安卓手机上部署。
 
这些实现都不是基于原训练框架实现。 Paddle-Lite提供了一整套完整的AI工具链，最终高效地部署到移动端推理框架。


# 五、设计思路与实现方案

项目基于PaddlePaddle论文复现开源项目[YOLOv5-Paddle](https://github.com/GuoQuanhao/YOLOv5-Paddle)做一些适当的修改.

完全参考Paddle-Lite-Demo中的目标检测项目(MobileNetV1-SSD, YOLOV3-MobileNetV3)，在此基础上替换掉模型，修改前后处理方式，UI和功能等将不做修改。


## 命名与参数设计

因为是报告形式，无 API 接口设计


# 六、测试和验收的考量

adb shell demo测试通过
可基于图片和视频流实现推理
CPU和GPU推理正确

# 七、可行性分析和排期规划

c++部分代码已经跑通，能推理出预期结果，在RCF验收通过后可快速提交。


# 八、影响面

单独的Demo项目，不影响推理框架以及其他Demo项目

# 名词解释

# 附件及参考资料
