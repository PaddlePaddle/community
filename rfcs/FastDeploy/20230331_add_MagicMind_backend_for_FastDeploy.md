# 实现MagicMind接入FastDeploy，并使用YOLOv5进行测试

| 方案名称                         |  MagicMind接入FastDeploy  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | MayYouBeProsperous                             | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-03-31                              | 
| 版本号                                                      | V1.0                                      | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==2.0                     | 
| 文件名                                                      | 20230331_add_MagicMind_backend_for_FastDeploy.md<br> | 

# 一、概述
## 1、相关背景
MagicMind 是面向寒武纪 MLU 的推理加速引擎。MagicMind 能将人工智能框架（TensorFlow、PyTorch、ONNX等）的算法模型转换成 MagicMind 统一计算图表示，并提供端到端的模型优化、代码生成以及推理业务部署能力。


## 2、功能目标
将 MagicMind 推理加速引擎接入 FastDeploy，实现 Paddle 到 向寒武纪 MLU 的快速部署。

## 3、意义
为 FastDeploy 完善对寒武纪 MLU 后端的支持，达到 FastDeploy 多端部署的产品目标。

# 二、寒武纪 MLU 开发现状

在寒武纪 MLU 开发时，开发者需要了解 MagicMind 的模型表示，模型生成配置，模型部署等内容，有一定的学习成本。

# 三、业内方案调研

目前 FastDeploy 已经支持较多的后端，开发者通过继承后端的基类，可以增加新的后端。

# 四、对比分析

FastDeploy 对后端的支持，主要体现在 `fastdeploy/runtime/backends` 目录中，不同的后端，需要分别实现各自的加载模型、获取输入输出、推理接口的函数。

此外 MagicMind 的专属接口，可根据实际情况，做相应的移植。

# 五、设计思路与实现方案

## 1、主体设计思路与折衷

### 主体设计具体描述
#### 后端继承的流程

1. 枚举类型声明

主要用于在 FastDeploy 中声明新加入的后端 MagicMind。

2. 后端接口实现

创建 `fastdeploy/runtime/backends/magicmind` 目录，并实现后端接口。

3. Backend 集成进 FastDeploy Runtime

实现统一的推理接口。

4. 编译CMake配置

创建 `FastDeploy/cmake/magicmind.cmake`，用于配置第三方库的下载，头文件的引入，以及库的引入。

修改 `FastDeploy/CMakeLists.txt`，添加`option(ENABLE_MAGICMIND)`、`file(GLOB_RECURSE DEPLOY_BACKEND_SRCS)`，`if(ENABLE_MAGICMIND)`的代码逻辑。

修改 `FastDeploy/FastDeploy.cmake.in`，在开始处获取编译参数，同时添加相应逻辑。

修改 `FastDeploy/fastdeploy/core/config.h.in`文件，加入宏定义。

5. C++后端测试

加载 YOLOv5 模型并推理测试。

6. 完成部署 demo

完成新后端的部署 demo，并撰写中英文文档。

## 2、关键技术点/子模块设计与实现方案
将 MagicMind 相关 API 集成进 Backends 是本次任务的核心部分，需要熟悉 MagicMind 的模型表示，模型生成配置，模型部署。任务计划以 ONNX 模型格式作为 MagicMind 引擎的输入格式，打通 FastDeploy 和 MagicMind。

## 3、主要影响的模块接口变化
在 `fastdeploy/runtime/backends` 中新增类，不影响 FastDeploy 已有后端。

# 六、测试和验收的考量
完成 benchmark 测试数据及精度对齐数据。

# 七、影响面
无

# 八、排期规划
* 2023-3-31 ~ 2023-4-15：完成集成代码开发
* 2023-4-16 ~ 2023-4-21：完成代码测试
* 2023-4-22 ~ 2023-4-26： 完成部署示例及文档

# 九、参考资料

[课程：FastDeploy接入寒武纪](https://aistudio.baidu.com/aistudio/education/lessonvideo/4132579)

[寒武纪 MagicMind ⽤户⼿册](https://www.cambricon.com/docs/sdk_1.10.0/magicmind_1.1.0/user_guide/index.html)

[MagicMind YOLOv5 c++部署实例](https://gitee.com/cambricon/magicmind_cloud/tree/master/buildin/cv/detection/yolov5_v6_1_pytorch)

[寒武纪 MagicMind C++ 开发者手册](https://www.cambricon.com/docs/sdk_1.10.0/magicmind_1.1.0/developer_guide/c++/index.html)

[MagicMind Benchmark指南](https://www.cambricon.com/docs/sdk_1.10.0/magicmind_1.1.0/performance_guide/index.html)


