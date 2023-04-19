# 完成CANN接入FastDeploy，并完成高性能文本分类服务ERNIE-3.0，测试模型运行正确

| 方案名称                         |  CANN接入FastDeploy  | 
|----------------------------------------------------------|-------------------------------------------|
| 提交作者<input type="checkbox" class="rowselector hidden">   | MayYouBeProsperous                             | 
| 提交时间<input type="checkbox" class="rowselector hidden">   | 2023-04-02                              | 
| 版本号                                                      | V1.0                                      | 
| 依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | paddlepaddle-gpu==2.0                     | 
| 文件名                                                      | 20230402_add_CANN_backend_for_FastDeploy.md<br> | 

# 一、概述
## 1、相关背景
CANN 是华为针对 AI 场景推出的异构计算架构，通过提供多层次的编程接口，支持用户快速构建基于 Ascend 平台的 AI 应用和业务。CANN 包含统一APP编程语言，提供了一套标准的Ascend CL 编程核口，提供模型加载与执行、媒体数据处理、算子加载与执行等API，能够实现在昇腾CANN平台上进行深度学习推理计算、图形图像预处理、单算子加速计算等能力。

## 2、功能目标
将 CANN 接入 FastDeploy，实现 Paddle 到 向异腾硬件的快速部署。

## 3、意义
为 FastDeploy 完善对异腾硬件的后端支持异构计算架构，达到 FastDeploy 多端部署的产品目标。

# 二、异腾硬件开发现状

在异腾硬件开发工作中，开发者需要了解 CANN 异构计算架构，熟悉 Ascend CL 昇腾计算语言，有一定的学习成本。

# 三、业内方案调研

目前 FastDeploy 已经支持较多的后端，开发者通过继承后端的基类，可以增加新的后端。

# 四、对比分析

FastDeploy 对后端的支持，主要体现在 `fastdeploy/runtime/backends` 目录中，不同的后端，需要分别实现各自的加载模型、获取输入输出、推理接口的函数。

# 五、设计思路与实现方案

## 1、主体设计思路与折衷

### 主体设计具体描述
#### 后端继承的流程

1. 枚举类型声明

主要用于在 FastDeploy 中声明新加入的后端 CANN。

2. 后端接口实现

创建 `fastdeploy/runtime/backends/cann` 目录，并实现后端接口。

3. Backend 集成进 FastDeploy Runtime

实现统一的推理接口。

4. 编译CMake配置

创建 `FastDeploy/cmake/cann.cmake`，用于配置第三方库的下载，头文件的引入，以及库的引入。

修改 `FastDeploy/CMakeLists.txt`，添加`option(ENABLE_CANN)`、`file(GLOB_RECURSE DEPLOY_BACKEND_SRCS)`，`if(ENABLE_CANN)`的代码逻辑。

修改 `FastDeploy/FastDeploy.cmake.in`，在开始处获取编译参数，同时添加相应逻辑。

修改 `FastDeploy/fastdeploy/core/config.h.in`文件，加入宏定义。

5. C++后端测试

加载 ERNIE-3.0 模型并推理测试。

6. 完成部署示例

完成新后端的部署 ERNIE-3.0 的demo，并撰写中英文文档。

## 2、关键技术点/子模块设计与实现方案
将 AscendCL 相关 API 集成进 Backends 是本次任务的核心部分，任务计划通过 ONNX 接入昇腾 AI 工具链。

## 3、主要影响的模块接口变化
在 `fastdeploy/runtime/backends` 中新增类，不影响 FastDeploy 已有后端。

# 六、测试和验收的考量
将验证通过的模型，提交到昇腾模型库；完成 benchmark 测试数据及精度对齐数据。

# 七、影响面
无

# 八、排期规划
* 2023-4-01 ~ 2023-4-16：完成集成代码开发
* 2023-4-17 ~ 2023-4-22：完成代码测试
* 2023-4-23 ~ 2023-4-30：完成部署示例及文档

# 九、参考资料

[课程：FastDeploy接入昇腾](https://aistudio.baidu.com/aistudio/education/lessonvideo/4132837)

[昇腾文档](https://www.hiascend.com/document)
