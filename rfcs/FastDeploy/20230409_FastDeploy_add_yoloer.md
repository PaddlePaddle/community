# 20230409_FastDeploy_add_yoloer 

| 提交作者     | 王夏兵、陈浩、刘彦涛              |
| ------------ | --------------------------------- |
| 提交时间     | 2023-04-09                        |
| 版本号       | V1.0                              |
| 依赖飞桨版本 | develop                           |
| 文件名       | 20230409_FastDeploy_add_yoloer.md |

# 一、概述

## 1、相关背景

​	使用过FastDeploy产品在边缘端部署过相关模型，了解模型的前后处理。目前，正在使用算丰平台进行嵌入式AI的开发，但是发现FastDeploy平台目前对算丰平台的模型的支持还太少，所以想借这个机会进行FastDeploy在算丰平台上的维护。

## 2、功能目标

​	完成整套API的开发，使用户仅使用FastDeploy提供的功能，就能够完成预期结果，而不用自己进行前后处理。

## 3、意义

​	完善硬件平台在FastDeploy上的API统一，使得用户能够更快得进行模型得部署。这个项目对于目前边缘端模型的部署有着极大的意义。

# 二、现状分析

- 飞桨框架下的FastDeploy项目目前并不支持对PP-YOLOE-R算法在算能1684和算能1684X的快速部署，其缺少对PP-YOLOE-R算法的前后处理的实现代码。

# 三、业内方案调研

- 目前业内还没有其他将PP-YOLOE-R算法模型前后处理以及模型推理集成在一起封装为SDK方便使用的方案。
- FastDeploy，将底层的硬件平台以及不同的操作系统抽象出来，使得用户在部署模型的时候可以不用过多考虑底层细节的东西。而且，其将模型的前后处理，以及模型推理集合在一起，封装成统一的API，使用户快速完成相关模型部署，直接完成端到端的部署。
- 目前，深度学习框架在支持算能部署方面的现状主要是基于TensorFlow Lite框架进行模型转换和优化。
- 各大深度学习框架均已开始探索在边缘设备上的部署和优化问题，例如TensorFlow Lite、PyTorch Mobile、Caffe2等。未来，随着边缘计算的发展和需求的增加，边缘AI芯片的性能和资源将逐渐提升，同时深度学习框架也将不断改进和优化，以更好地支持在边缘设备上部署深度学习模型。

# 四、对比分析

- 目前业内还没有其他类似FastDeploy的统一边缘端的方案。

# 五、设计思路与实现方案

## 命名与参数设计

​	完全参考其他模型处理的命名规范。

## API设计

preprocessor.h: 前处理

```c++
class FASTDEPLOY_DECL YoloerPreprocessor {
    private:
        bool BuildPreprocessPipelineFromConfig();//初始化配置文件
        std::vector<std::shared_ptr<Processor>> processors_;//记录要进行哪些处理
        bool initialized_ = false;
        bool disable_permute_ = false;
        bool disable_normalize_ = false;
        std::string config_file_;//配置文件路径
  	protected:
        bool Run(...);//主要逻辑逻辑实现
        void LetterBox(...);//图片处理    
    public:
        YoloerPreprocessor(...);//初始化
        bool Preprocess(...);//前处理
	...
};
```

postprocessor.h: 后处理

```c++
class FASTDEPLOY_DECL YoloerPostprocessor {
	protected:
    	std::vector<float> scale_factor_{0.0, 0.0};
        bool Run(...);//主要逻辑实现
    public:
        YoloerPostprocessor(...);//初始化
        bool Postprocess(...);//后处理主函数
        ...
};
```

yoloer.h: 与用户交互的主要函数，完成模型部署主要功能的实现

```c++
class FASTDEPLOY_DECL YOLOER : public FastDeployModel {
	protected:
        YoloerPreprocessor preprocessor_;
        YoloerPostprocessor postprocessor_;
    public:
        YOLOER(...);//初始化
        virtual bool Predict(const std::vector<cv::Mat>& images, 					   std::vector<FaceDetectionResult>* result); //完成对输入的预测
        virtual YoloerPreprocessor& GetPreprocessor();
        virtual YoloerPostprocessor& GetPostprocessor(); 
        virtual std::string ModelName();
    ...
};
```



# 六、测试和验收的考量

参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

- 考虑API单测开发规范，只采用动态图的编程范式，应该涵盖测试参数组合场景以及其他异常测试。
- 考虑API单测通用规范，保证新增API的命名规范，提交规范，覆盖率规范以及耗时规范。

- 验证在CPU，GPU环境下使用封装的SDK快速部署后的效果

- 验证在算能1684，算能1684x的部署后的精度，以及速度测试。


# 七、可行性分析和排期规划

在比赛结束前完成。

1. 完成代码开发
2. 提交使用案例
3. 测试模型部署功能

# 八、影响面

- 由于是针对单独算法的新增API的设计，不涉及其他模块的改动，对其他模块没有影响
- 所有代码均在 `fastdeploy/vision/detection/ppdet`目录下，对其他模块没有影响。
