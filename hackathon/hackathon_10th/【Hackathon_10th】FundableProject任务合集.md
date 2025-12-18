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
2. Open3D: [https://www.open3d.org/docs/release/compilation.html](https://www.open3d.org/docs/release/compilation.html)

