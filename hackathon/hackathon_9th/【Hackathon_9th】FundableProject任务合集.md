此文档展示 **PaddlePaddle Hackathon 第九期活动——Fundable Projects** 任务详细介绍。Fundable Projects 赛道定位硬核任务，要求高水平的开发者独立进行任务拆解和完成。

## 产出要求

- 任务拆解 tracking issue
- 答辩 PPT
- 书面的技术报告
- 代码运行无误，通过社区 maintainers 的评审并合入代码仓库。

## 任务详情

### 一、第三方生态库pytorch_cluster适配

**任务背景**：

pytorch_cluster包含用于PyTorch的高度优化的图聚类算法，是知名图神经网络库pytorch_geometric底层基础库之一，为了完整实现pytorch_geometric需要实现实现pytorch_cluster。

**详细描述：**

1. 完整实现pytorch_cluster中的全部API，精度实现对齐；
2. 实现对应的单元测试；
3. 该项目要求 9 月底收尾，无法保证完成时间的开发者不建议领取。

**验收说明：**

1. 实现pytorch_cluster中的API，并给出安装、使用说明；
2. 实现 其中对应的单元测试并通过；
3. 最终代码合入PFCCLab组织下；

**技术要求：**

- 熟悉 Python，C++，工程能力强
- 对PyTorch、Paddle比较熟悉

**参考资料：** pytorch_cluster: [https://github.com/rusty1s/pytorch_cluster](https://github.com/rusty1s/pytorch_cluster)

### 二、第三方生态库pytorch_spline_conv适配

**任务背景**：

pytorch_spline_conv实现了一种基于B-splines的新型卷积算子，用于不规则结构和几何输入的深度神经网络变体，例如图形或网格。pytorch_spline_conv是知名图神经网络库pytorch_geometric底层基础库之一，为了完整实现pytorch_geometric需要实现实现pytorch_spline_conv。

**详细描述：**

1. 完整实现pytorch_spline_conv中的全部API，精度实现对齐；
2. 实现对应的单元测试；
3. 该项目要求 9 月底收尾，无法保证完成时间的开发者不建议领取。

**验收说明：**

1. 实现pytorch_spline_conv中的API，并给出安装、使用说明；
2. 实现 其中对应的单元测试并通过；
3. 最终代码合入PFCCLab组织下；

**技术要求：**

- 熟悉 Python，C++，工程能力强
- 对PyTorch、Paddle比较熟悉

**参考资料：**

1. pytorch_spline_conv: [https://github.com/rusty1s/pytorch_spline_conv](https://github.com/rusty1s/pytorch_spline_conv)

### 三、工具组件适配dgl适配

**任务背景**：

现有 Paddle 3.0 版本基础功能基本持平 PyTorch，在部分功能上领先 PyTorch，但在外部合作、内部研发过程中发现，框架代码上层的配套基础设施缺失，DGL 就是其中之一。

DGL 是一个易于使用、高性能和可扩展的 Python 包，用于图形深度学习。它是框架无关的，这意味着如果深度图模型是端到端应用程序的一个组件，那么其余的逻辑可以在任何主要框架中实现。

它当前支持的后端包括：PyTorch、TensorFlow 和 Apache MXNet。

**详细描述：**

1. 参考“参考资料”部分，修改、新增 dgl/python/ 目录下代码，为 DGL 适配 Paddle 后端，实现当前 TensorFlow 后端中支持的模型/算子/优化器等；
2. 实现当前 PyTorch 后端中支持的模型/算子/优化器等，主要实现 PyTorch 后端中对稀疏类型进行处理的部分，如dgl/python/dgl/optim/pytorch/sparse_optim.py 和 dgl/python/dgl/nn/pytorch/conv/cugraph_gatconv.py 等；
3. 实现对应的单元测试；

**验收说明：**

1. 与 TensorFlow 后端支持等量功能；
2. 实现 1 中对应的单元测试并通过；
3. 与 PyTorch 后端支持等量功能；
4. 实现 3 中对应的单元测试并通过；
5. PR 合入 DGL 仓库；

**技术要求：**

- 熟悉 Python，工程能力强
- 对 PyTorch 或 TensorFlow 比较熟悉
- 对 DGL 套件比较熟悉（加分项）

**参考资料：**

1. dgl 仓库: https://github.com/dmlc/dgl/tree/master
2. 当前已有 paddle backend 适配实现: https://github.com/lijialin03/dgl/tree/bkd_paddle

### 四、动转静 SOT Python 3.14 支持

**任务背景**：

动转静 SOT 模块是基于 Python 字节码的 JIT 编译模块，旨在在运行时将 PaddlePaddle 动态图组网代码转换为静态图组网代码，具体设计参见：[PaddleSOT 项目介绍](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/symbolic_opcode_translator)
由于 SOT 模块需要针对每条字节码进行模拟执行，并在 CodeGen 时生成合法的 Python 字节码，因此对于 Python 版本非常敏感。我们现在对 Python 3.9-3.13 已经有了较为全面的支持，但新发布的 Python 3.14 目前还是不支持的，因此需要专项对 Python 3.14 进行支持。

**详细描述：**

1. 参考 [Python 3.11 支持规划](https://github.com/PaddlePaddle/PaddleSOT/issues/357)、[SOT Python3.12 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/61173)、[SOT Python 3.13 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/69245)，调研 Python 3.14 主要改动，确定 Python 3.14 支持路线
2. PR-CI-SOT 流水线上线 Python 3.14 监控，确保已有单测不会回归
3. 适配 Eval Frame 模块，适配模拟执行、CodeGen 等流程

**验收说明：**

1. CI 流水线能够监控 Python 3.14 SOT 单测
2. SOT 在 Python 3.14 下功能完备，全部 SOT 单测能够在 Python 3.14 下验证通过

**技术要求：**

- 精通 Python，对 Python 虚拟机执行机制有深入了解
- 熟悉 C/C++
- 掌握基本的编译原理知识

**参考资料：**

1. [Python 3.11 支持规划](https://github.com/PaddlePaddle/PaddleSOT/issues/357)
2. [SOT Python3.12 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/61173)
3. [SOT Python 3.13 支持任务汇总](https://github.com/PaddlePaddle/Paddle/issues/69245)
