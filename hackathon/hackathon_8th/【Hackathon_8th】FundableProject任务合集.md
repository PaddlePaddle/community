此文档展示 **PaddlePaddle Hackathon 第八期活动——Fundable Projects** 任务详细介绍。Fundable Projects 赛道定位硬核任务，要求高水平的开发者独立进行任务拆解和完成。

## 产出要求

- 任务拆解 tracking issue
- 答辩 PPT
- 书面的技术报告
- 代码运行无误，通过社区 maintainers 的评审并合入代码仓库。

## 任务详情

### 一、PaddleSpeech 套件能力建设 - 模型精度对齐

**任务背景**：

PaddleSpeech 是基于飞桨 PaddlePaddle 的语音方向的开源套件，囊括语音识别、语音合成、语音唤醒、声纹识别等多种语音常用功能的支持。由于近期 Paddle 新版本的升级存在不兼容部分（如 `paddle.fluid` API 全面退场，PIR + predictor 升级， 0-d tensor，view 行为修改等），需要重新对 PaddleSpeech 中的模型进行适配开发与回归测试，保证套件正常运转，模型功能与精度不受损失。外部开发者需要做的事情包括：

**详细描述：**

基于 Paddle 3.0.0-beta 版本对 PaddleSpeech 进行适配升级，梳理已有堵点并解决。保证[example](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples) 目录下核心模型在 新 Paddle 版本 & 新其他深度学习框架版本下的正常运转。目前适配版本为 Paddle 2.5.1。

**验收说明：**

PaddleSpeech 基于 Paddle 3.0.0-beta 版本，完成 10+ 原有模型的适配和精度对齐。

**技术要求：**

- 熟悉 Python，工程能力强
- 对语音识别或合成有一定了解，有训练或者研发经验（加分项）
- 对 PaddleSpeech 套件比较熟悉（加分项）

**参考资料：** https://github.com/PaddlePaddle/PaddleSpeech

### 二、PaddleSpeech 套件能力建设 - PIR 导出

**任务背景**：

PaddleSpeech 是基于飞桨 PaddlePaddle 的语音方向的开源套件，囊括语音识别、语音合成、语音唤醒、声纹识别等多种语音常用功能的支持。由于近期 Paddle 新版本的升级存在不兼容部分（如 `paddle.fluid` API 全面退场，PIR + predictor 升级， 0-d tensor，view 行为修改等），需要重新对 PaddleSpeech 中的模型进行适配开发与回归测试，保证套件正常运转，模型功能与精度不受损失。外部开发者需要做的事情包括：

**详细描述：**

基于 Paddle 3.0.0-beta 版本对 PaddleSpeech 中支持转静的模型重新按照 PIR + predictor 的方式导出，并成功推理。

**验收说明：**

PaddleSpeech 基于 Paddle 3.0.0-beta 版本，完成 20+ 原有静态图模型的重新导出和上传。

**技术要求：**

- 熟悉 Python，工程能力强
- 对语音识别或合成有一定了解，有训练或者研发经验（加分项）
- 对 PaddleSpeech 套件比较熟悉（加分项）

**参考资料：** https://github.com/PaddlePaddle/PaddleSpeech

### 三、科学计算方向开源工具组件适配 - Paddle 后端适配

**任务背景**：

现有 Paddle 3.0 版本基础功能基本持平 PyTorch，在部分功能上领先 PyTorch，但在外部合作、内部研发过程中发现，框架代码上层的配套基础设施缺失，DGL 就是其中之一。

DGL 是一个易于使用、高性能和可扩展的 Python 包，用于图形深度学习。它是框架无关的，这意味着如果深度图模型是端到端应用程序的一个组件，那么其余的逻辑可以在任何主要框架中实现。

它当前支持的后端包括：PyTorch、TensorFlow 和 Apache MXNet。

**详细描述：**

1. 参考“参考资料”部分，修改、新增 dgl/python/ 目录下代码，为 DGL 适配 Paddle 后端，实现当前 TensorFlow 后端中支持的模型/算子/优化器等；
2. 实现当前 PyTorch 后端中支持的模型/算子/优化器等，主要实现 PyTorch 后端中对稀疏类型进行处理的部分，如 dgl/python/dgl/optim/pytorch/sparse_optim.py 和 dgl/python/dgl/nn/pytorch/conv/cugraph_gatconv.py 等；
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

### 四、科学计算方向开源工具组件 pytorch_sparse 适配

**任务背景：**

[pytorch_sparse](https://github.com/rusty1s/pytorch_sparse)是一个专门为 PyTorch 框架设计的扩展库，它提供了对稀疏张量的高效操作和优化。稀疏张量在处理具有大量零值的数据时非常有用，能够显著减少内存占用和提高计算效率。PaddlePaddle 支持稀疏 Tensor，但是与该库的功能仍有较大差距。因此为了扩充 PaddlePaddle 对稀疏矩阵的支持，对 pytorch_sparse 基于 PaddlePaddle 进行实现。

**任务描述：**

1. 实现 SparseTensor 类及其成员函数，另外需要实现该类的 share*memory*、is_shared、to、cpu、cuda、**getitem**、**repr**、from_scipy、to_scipy、index_select、index_select_nnz、masked_select、masked_select_nnz 等函数，实现精度与性能对齐；
2. 实现 SparseStorage 类及其成员函数，另外需要实现该类的 share_memor\_、is_shared 等函数，实现精度与性能对齐；
3. 实现对应的单元测试；

**验收说明：**

1. 实现 SparseTensor 及其相关成员函数；
2. 实现 SparseStorage 类及其相关成员函数；
3. 性能与精度与 Torch 对齐，提供对齐代码及结果；
4. 实现 其中对应的单元测试并通过；
5. 最终代码合入 PFCCLab 组织下；

**技术要求：**

- 熟悉 Python，C++，工程能力强
- 对 PyTorch、Paddle 比较熟悉

**参考资料：**

1. pytorch_sparse: https://github.com/rusty1s/pytorch_sparse
2. paddle_scatter: https://github.com/PFCCLab/paddle_scatter
