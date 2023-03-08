# Paddle Frawework Contributor Club 第十七次会议纪要

## 会议概况

- 本次会议时间：2022-03-02 19:00
- 会议地点：线上会议
- 参会人：本次会议共有38名成员参会，由来自全国各地的飞桨框架的贡献者组成。本次会议由轮值主席卢雨畋（[sanbuphy](https://github.com/sanbuphy)）主持。
- 会议主题：《飞桨 TRT 开发与 inference GPU 适配开发、飞桨 fx 构思分享》


## 会议分享与讨论

本次会议主要以飞桨 TRT 算子开发为主题进行相关内容的分享与讨论。

以下为主要内容：

### 1、新成员的自我介绍
首先，PFCC 新成员 [CollaborativeFiltering](https://github.com/CollaborativeFiltering)、[edencfc](https://github.com/edencfc) 、[cloud2009](https://github.com/cloud2009) 进行了自我介绍，欢迎加入 PFCC！

### 2、飞桨 TRT 开发经验

飞桨研发[zhangjun](https://github.com/zhangjun)，主要分享飞桨 TRT 开发经验，详细介绍了常见的飞桨 RT 扩展增加方案与 RT 扩展执行流程，并对 RT 映射做了简单的介绍。

### 3、飞桨 TRT 算子映射开发经验分享

PFCC成员[sanbuphy](https://github.com/sanbuphy) 针对几个 PR 进行了飞桨 TRT 算子映射开发流程分享。

### 4、Python IR 构建实验介绍

PFCC成员[jzhang533](https://github.com/jzhang533) 进行了 Python IR 构建实验介绍。目标是以可编程的方式基于这个IR做变换，简化，以适用到其他场景。（如特征提取、性能优化等）之后可以基于这个IR 将paddle模型lower到其他的执行引擎上执行。

具体项目：https://github.com/PFCCLab/paddlefx
简介：https://github.com/PFCCLab/paddlefx/discussions/12

### 5、飞桨 inference 对于非cuda-like平台的gpu适配方法

PFCC成员[engineer1109](https://github.com/engineer1109) 讲解了有关飞桨 inference 对于非cuda-like平台的gpu适配方法，主要涉及到 Paddle Inference 完成适配需要的接口、OpenCL内存池模型、OpenCL SVM共享虚拟内存、Vulkan、OpenGL等方案。


### 6、黑客松活动同步

飞桨产品 [Krystalxxf](https://github.com/Krystalxxf) 进行了社区信息同步，介绍最新的开源活动——飞桨黑客松第四期。


### 下次会议安排

确定下次会议的时间为两周后的同一个时间段。主席为[gouzil](https://github.com/gouzil), 副主席[待定]()，主题待定。
