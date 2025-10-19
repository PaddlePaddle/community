此文档展示 **PaddlePaddle Hackathon 第八期活动——飞桨护航计划集训营（正式批）** 项目的详细介绍

## 赛题详情

### 项目一：框架 API 易用性提升

#### 项目介绍：

飞桨框架 API 中 **功能缺失、功能 Bug** 等地方，都将成为 Pytorch 代码转换的堵点，为了扫清**框架 API 转换**的堵点，**降低模型迁移的成本**，需要对框架 API 的易用性进行提升。

每一个 API 任务包括：问题描述 + 建议方案，其中 **问题描述** 会给出该 API 存在的不合理问题，**建议方案** 会给出该 API 的建议修改思路（例如 功能增强、Bug 修复、实现优化、新增 API 等），你需要参考**建议方案** ，完成该 API 的增强或新增工作。

除了对框架 API 本身的修改外，还需要完成 [PaConvert 代码转换](https://github.com/PaddlePaddle/PaConvert)、[API 映射文档](https://github.com/PaddlePaddle/docs/blob/develop/docs/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_format_cn.md) 的开发。因此你需要额外熟悉 [PaConvert](https://github.com/PaddlePaddle/PaConvert) 仓库，保证 API 转换的单测可以成功运行。

每一个 API 任务总计包括如下开发内容：（注意逐个核对是否完成）

1. **API 本身修改**：代码提交至 [Paddle 仓库](https://github.com/PaddlePaddle/Paddle)
2. **API 英文、中文文档修改**：代码分别提交至 [Paddle 仓库](https://github.com/PaddlePaddle/Paddle) 和 [docs 仓库](https://github.com/PaddlePaddle/docs)
3. **API 映射文档修改**：代码提交至 [映射文档目录](https://github.com/PaddlePaddle/docs/tree/develop/docs/guides/model_convert/convert_from_pytorch/api_difference/)
4. **PaConvert 转换规则修改**：代码提交至 [PaConvert 仓库](https://github.com/PaddlePaddle/PaConvert)
5. **PaConvert 单测修改**：代码提交至 [PaConvert 单侧目录](https://github.com/PaddlePaddle/PaConvert/tree/master/tests)

具体待升级的 API 任务名单，以内部实际发布为准，在结营时要求完成 10~20 个 API 升级任务。

#### 营员要求（1 人）：

- 熟悉 Python
- 熟悉 C++、CUDA
- 熟悉飞桨 API 算子开发，有新增 API 经验者优先
- 熟悉 Pytorch 框架的使用，有论文复现与模型迁移经验优先

### 项目二：编译器后端架构优化专项

#### 项目介绍：

飞桨 3.0 推出了与框架一体化的 CINN 编译器，能将高层次的深度学习模型转换为低层次的、高效的、底层硬件可执行的代码。目前，CINN 的编译器在架构上主要分为前端和后端。前端主要完成一些图层优化，后端主要是根据目标硬件的特性进行进一步的优化与代码生成。然而，优越的性能表现和通用性也伴随着架构的复杂，目前 CINN 后端存在编译耗时高，耦合重等问题。除此之外，多种多样的优化方法间也可能存在冲突，需要合理调度。

本专项课题期望选手参与飞桨编译器后端建设，根据项目需求完成对后端各模块的架构升级，并对各种优化方法进行分析与调度优化，提升编译器的可维护性与可扩展性。

#### 营员要求（1-2 人）：

- 熟悉 C++
- 了解 CUDA 编程与优化（加分项）
- 了解深度学习编译技术（加分项）

### 项目三：混合专家结构自动切分推导优化

#### 项目介绍：

飞桨自动并行架构通过对分布式计算任务和硬件资源的统一抽象，让用户可以通过简单的张量切分语法标记实现不同的并行策略，底层由框架自动推导出所有张量和算子的分布式切分状态，并自动添加合适的通信操作。自动推导张量的分布式切分状态需要使用网络中算子的切分推导规则，切分推导的完备性和合理性将直接影响推导出的模型切分状态，进而影响并行和通信的效率。近来，混合专家（Mixture of Experts, MoE）结构受到越来越高的关注，MoE 模型中常用的一些算子，例如 einsum、top_k 等，在框架中暂未有合适的切分推导实现，使得自动并行在 MoE 场景下难以自动推导出较优的切分状态。本课题要求选手系统地学习和理解 MoE 网络结构及自动并行的切分推导机制，梳理出影响 MoE 模型切分推导的算子，并为这些算子实现合理的切分推导规则，使得自动并行在 MoE 场景下能自动推导出较优的切分状态，提升架构性能。

#### 营员要求（1 人）：

- 熟悉 Python、C/C++
- 熟悉深度学习框架中的动态图和静态图
- 熟悉自动并行相关基础概念

### 项目五：框架作为 array-api-compat 后端

#### 项目介绍：

适配 paddle 的 API，满足 array-api-compat 的测试要求，并且将 paddle 作为后端合入 array-api-compat

主要工作内容：

- 适配 paddle 的 API，满足 array-api-compat 的测试要求，并且将 paddle 作为后端合入 array-api-compat

#### 营员要求（1 人）：

- 熟悉 Python
- 掌握基本的深度学习框架使用能力

### 项目六：高效表格识别新范式探索

#### 项目介绍：

表格识别任务旨在将表格图片转换为 HTML、Excel 等结构化格式，并保持表格结构、表格内容等信息与图片中表格一致，是 PaddleX 为用户提供的最重要特色能力之一，也是目前学术界、产业界的研究难点之一。因此，我们期望在 PaddleX 已有方案（表格识别产线、表格识别v2产线）的基础上设计一种更高效的表格识别新范式。

主要工作内容：

- 基于 PaddleX 提供的表格单元格检测模型，探究基于其预测结果做后处理以实现高效表格图片结构化的策略
- 基于新策略开发高效表格识别系统，并将符合要求的代码合入 PaddleX 仓库中

#### 营员要求（1 人）：

- 熟悉 Python，编程经验良好
- 了解飞桨 PaddlePaddle 框架的使用，能够使用飞桨进行数据处理、模型预测等
- 熟悉计算机视觉、OCR、文档解析等任务基础知识和前沿算法，有相关领域论文发表者优先
- 有表格识别、文档结构化等方向研究或实战经验者优先
