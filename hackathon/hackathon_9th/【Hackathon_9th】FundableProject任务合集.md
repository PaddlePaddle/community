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