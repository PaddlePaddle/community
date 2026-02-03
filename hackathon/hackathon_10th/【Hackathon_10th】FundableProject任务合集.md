此文档展示 **PaddlePaddle Hackathon 第十期活动——Fundable Projects** 任务详细介绍。Fundable Projects 赛道定位硬核任务，要求高水平的开发者独立进行任务拆解和完成。

## 产出要求

- 任务拆解 tracking issue
- 答辩 PPT
- 书面的技术报告
- 代码运行无误，通过社区 maintainers 的评审并合入代码仓库。

## 任务详情

### 一、科学计算方向开源工具套件PaddleCFD在海光DCU环境适配

**任务背景**：

PaddleCFD基于自定义算子编译ppfno_op，提供了强大的流体力学仿真功能。该套件依赖于Open3D等第三方库。但目前，该依赖主要仅实现了x86+CUDA环境的安装支持。为了扩充PaddleCFD对海光DCU环境的支持（whl等），需要对Open3D等进行单独适配，从而支持PaddleCFD的安装和运行。

**详细描述：**

1. 实现Open3D、PaddleCFD对海光DCU环境的安装运行支持；
2. 实现对应的Benchmark测试。

**验收说明：**

1. 实现PaddleCFD在海光DCU环境的安装，提供whl文件；
2. Benchmark性能与精度与CUDA环境对齐，提供对齐代码及结果；
3. 用户友好的安装说明文档；
4. 最终代码合入PaddleCFD下。

**技术要求：**

- 熟悉Python，C++，工程能力强；
- 熟悉Paddle、Hygon技术栈。

**参考资料：** 

1. PaddleCFD: [https://github.com/PaddlePaddle/PaddleCFD](https://github.com/PaddlePaddle/PaddleCFD)
2. Open3D: [https://github.com/PFCCLab/Open3D](https://github.com/PFCCLab/Open3D)

### 二、FastDeploy缺陷检测与修复

**任务背景**：

基于值依赖分析技术和大语言模型RAG技术，对飞桨 [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) 推理框架进行业务缺陷检测与通用缺陷检测，能够发现传统技术难以分析的缺陷，如深度学习框架代码理解、函数语义理解等，并且实现修复，能有效解决了深度学习推理框架代码在业务逻辑层面难以被传统分析手段检测的问题，大大提高了飞桨推理框架的安全性和可靠性。

**详细描述：**

1. 对飞桨 FastDeploy 推理框架进行代码缺陷检测与定位。
2. 对通过人工审查，可以确定为真实缺陷的结果列表，进行修复（包括自动、手动修复）。

**验收说明：**

1. 提供易读的缺陷结果列表，包含缺陷类别、所在文件路径与行号、缺陷说明等。
2. 对通过人工审查，可以确定为真实缺陷的结果列表，进行修复，代码合入FastDeploy下。
3. PR描述中包含修复说明，可参考[【Paddle仓库下的缺陷修复案例】](https://github.com/PaddlePaddle/Paddle/pulls?q=is%3Apr+label%3A%22Beijing+Innovation+Consortium%22+is%3Aclosed)。
4. 奖励说明：每条确定为真实缺陷且修复代码合入仓库，获得0.05⭐️，本课题封顶10x⭐️。

**技术要求：**

- 熟悉Python，C++，工程能力强
