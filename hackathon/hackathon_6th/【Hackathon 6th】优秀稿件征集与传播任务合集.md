此文档展示 **PaddlePaddle Hackathon 第六期活动——优秀稿件征集与传播** 的任务详细介绍。选手可以写 **选定知识点内** 的任意有助于扩大飞桨影响力的文章，更多详见 [PaddlePaddle Hackathon 说明]()。

## 任务详情

### NO.1 Inplace 的使用指南&学习心得

**详细要求：**

稿件均需结合实际 paddle 代码的例子进行说明，可选择知识点：

1. paddle 的 inplace 使用指南
2. 为什么 paddle 会需要 inplace & 使用 inplace 有什么好处？
3. 介绍 paddle 是如何实现 inplace 的？
4. inplace 在训练中会有哪些问题& paddle 是如何解决这些问题的？

**参考文档：**

- [飞桨 Inplace 介绍 & 使用介绍](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/Inplace/inplace_introduction.md)
- **Release/2.6 NOTE**：飞桨对 Inplace 机制进行了全面升级。实现自动检测反向计算对前向输入的依赖关系，并在需要时保存这些输入数据，从而支持更多的 Inplace 操作。这一改进不仅提升了内存使用效率，还增强了 API 的功能性。我们新增了 109 个支持 Inplace 操作的 API
- **Release/2.1 NOTE**: 新增可降低显存占用与提升性能的 inplace 操作，包括 View 策略，与 12 个 inplace API。

### NO.2 复数计算的使用指南&学习心得

**详细要求：**

稿件均需结合实际 paddle 代码的例子进行说明，可选择知识点：

1. paddle 的复数梯度是如何计算的？
2. paddle 的复梯度在链式法则中是如何体现的？
3. paddle 计算出来的梯度为什么是共轭梯度？

**参考文档：**

- [复数梯度推导计算](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/complex_autograd)
- [paddlepaddle 支持复数任务](https://github.com/PaddlePaddle/Paddle/issues/61975)
- [On the Computation of Complex-valued Gradients with Application to Statistically Optimum Beamforming](https://arxiv.org/abs/1701.00392)
- **Release/2.0 NOTE:** 扩展动静态图执行引擎，支持复数神经网络训练与复数梯度累加。新增 mul, div, matmul, kron, abs 等 Op 对复数计算支持。

### NO.3 Tensor 索引的使用指南&学习心得

**详细要求：**

稿件均需结合实际 paddle 代码的例子进行说明，可选择知识点：

1. 索引的概念、功能和使用场景
2. 索引的梯度计算规则
3. 结合几种不同领域的具体模型代码，介绍相关的索引在其中的作用

**参考文档：**

- https://github.com/PaddlePaddle/docs/pull/6507/files
- **Release/2.6 NOTE**：在基础索引中增加了对 view 的支持，修正了高级索引中的一些错误行为，并实现了联合索引的读取功能。此外，我们还将索引解析下沉到 C++层面，改进了高级索引算子的性能，并移除了 bool 索引中的冗余计算。
- **Release/2.2 NOTE**：新增多种索引类型的支持，新增的索引类型包括：省略号（…）、维度扩增（None）、布尔类型数组（Bool Mask）、整数数组(（list)，以及张量（Tensor）），可以更加方便的对张量（Tensor）进行操作。

### NO.4 线性代数、傅里叶变换、概率分布的使用指南&学习心得

**详细要求：**

稿件均需结合实际 paddle 代码的例子进行说明，领域应用建议通过一个应用案例贯通应用场景 & 涉及到的技术介绍，可选择知识点：

- 飞桨线形代数基础及领域应用
- 飞桨概率分布基础及领域应用
- 飞桨傅里叶变换基础及领域应用

**参考文档：**

- 官网有 [paddle.linalg](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/Overview_cn.html)、 [paddle.fft](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fft/Overview_cn.html)、[paddle.distribution](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/distribution/Overview_cn.html) 目录
- **Release/2.2 NOTE:** 新增 100+个 API，包含 24 个傅里叶变换 API、14 个线性代数计算 API 等，更好地支持科学计算类、信号处理类模型。

### NO.5 高阶微分的使用指南&学习心得

**详细要求：**

稿件均需结合实际 paddle 代码的例子进行说明，可选择知识点：

- 飞桨高阶自动微分在微分方程求解中应用

**参考文档：**

- 应用文档 https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/deephpms/

### NO.6 稀疏计算的使用指南&学习心得

**详细要求：**

稿件均需结合实际 paddle 代码的例子进行说明，可选择知识点：

1. 稀疏格式介绍以及 Paddle 支持哪些稀疏格式
2. Paddle 的稀疏调用体验与稠密高度一致，容易上手，并举几个例子（稀疏 ResNet 等，可自行编写简单例子）
3. Padlde 支持了哪些稀疏计算，支持的经典神经网络用法，举一些例子（3D 点云/Sparse Transformer）

**参考文档：**

- 官网有 [paddle.sparse](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/sparse/Overview_cn.html) 目录
- 公众号文章：[飞桨框架 v2.4 API 新升级！全面支持稀疏计算、图学习和语音处理等任务](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/HUnfvNS358/oTKIWDjqifSCI_)
- **Release/2.4 NOTE**：新增 55 个稀疏 API `paddle.sparse.*`，支持稀疏计算主流场景，已应用于 3D 点云目标检测、Sparse Transformers 等任务的稀疏训练和推理部署，高稀疏度场景下相比使用 DenseTensor 提速 105.75%，相比同类产品稀疏计算提速 4.01%~58.55%；支持多种稀疏 Tensor(SparseCoo 和 SparseCsr 等)的计算，极致节省显存；同时保持了一致的使用体验，和稠密 Tensor 的 API 使用方式一致。

### NO.7 0 维 tensor 的使用指南&学习心得

**详细要求：**

稿件均需结合实际 paddle 代码的例子进行说明，可选择知识点：

1. 0 维 Tensor 的概念，从数学与物理上与 1 维 Tensor（shape 为[1]）进行概念对比，讲解清晰
2. 0 维 Tensor 滥用为 1 维 Tensor（shape 为[1]）的危害，对体验的影响，列举一些例子
3. 对于深度学习框架，在哪些情况下应支持 0 维（至少 9 种情形），以及 Paddle 的支持情况，列举一些例子
4. 对比竞品 Pytorch 与 Paddle，并列举一些 Paddle 表现上更好的例子，从语义上说明 Paddle 的更合理性

   - Paddle.upsample 的 scale_factor 系数可支持 float/Tensor，而竞品仅支持 float，所以 paddle 相比 torch 额外多支持 0D
   - torch 的所有 loss 函数均支持 input、label 为 0D 输入，但 input、label 一般情况下都是有维度的，因此 paddle 不支持 0D

**参考文档：**

- [飞桨 0 维 Tensor 相关材料](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/ZeroDim)
- **Release/2.5 NOTE**：飞桨 API 支持 0 维 tensor。飞桨之前用 shape 为[1]的 1 维 tensor 来替代 0 维 tensor，这种替代方式和当前主流习惯有差异，增加模型的开发调试成本，有时还会导致非预期错误。本版本对需支持 0 维 tensor 的 376 个 API 进行了修正，和社区广泛使用的工具如 EinOps 等实现。
