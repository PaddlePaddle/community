此文档展示 **PaddlePaddle Hackathon 第八期活动——飞桨护航计划集训营（提前批）** 项目的详细介绍

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

#### 营员要求（1-2 人）：

- 熟悉 Python
- 熟悉 C++、CUDA
- 熟悉飞桨 API 算子开发，有新增 API 经验者优先
- 熟悉 Pytorch 框架的使用，有论文复现与模型迁移经验优先

### 项目二：模型迁移工具建设

#### 项目介绍：

为了实现高效的将 PyTorch 代码自动化的转写成 Paddle 代码，从而提升模型迁移的效率，我们建设了[**PaConvert 代码自动转换工具**](https://github.com/PaddlePaddle/PaConvert): **Pa**ddlePaddle Code **Convert** Toolkits。目前已支持约 1400 个 Pytorch API 的自动转换与 90+%的代码转换率，但在 新增 API 转写策略、转换机制优化、辅助函数优化、CI 维护与优化 等方面，仍然有很多可持续完善的地方。

本课题的工作任务包括转换工具建设的以下内容：

1. 新增 API 转写策略（约 150 个）
2. 转换机制优化（解耦 import 分析，从而支持 Comate 中任意选中代码的转换）
3. 辅助函数优化
4. CI 维护与优化（日常修复 CI 问题，增强存量问题的 CI 拦截）

#### 营员要求（1 人）：

- 熟悉 Python
- 熟悉 Pytorch 框架的使用，有论文复现与模型迁移经验优先

### 项目三：自动并行切分转换和专家并行机制完善

#### 项目介绍：

自动并行可在用户仅标记了模型中部分张量在不同进程上的分布情况后，自动实现高效的分布式训练。用户在使用时只需对部分张量进行标记，无需实现模型并行时的通信等流程，因此可以大幅简化分布式训练的开发流程。Paddle 的自动并行机制可以支持多种常见的分布式训练策略，但对于混合专家模型的专家并行策略还有待完善。

本课题包括以下内容：

1. 完善自动并行中的切分转换机制，更好地支持专家并行以及其他分布式策略。
2. 完善专家并行机制，提升框架在专家并行中的易用性，并在相关开源模型上验证。

#### 营员要求（1 人）：

- 熟悉 Python、C/C++
- 熟悉深度学习框架中的动态图和静态图
- 熟悉自动并行相关基础概念

### 项目四：静态图流水并行功能增强和性能优化

#### 项目介绍：

流水并行（Pipeline Parallel）是大模型分布式训练场景中非常重要的一种并行策略，它允许将模型的执行过程进行切分，使得多个微批次（micro-batch）可以同时执行模型代码的不同部分，以流水线的形式并发执行。飞桨自动并行上层以 pass 化的方式进行流水子图的调度编排，底层由执行器根据上层编排顺序进行多 micro-batch 的展开和执行，这种“先编排后执行”的设计方式允许根据实际场景的需要灵活地编排和扩展不同的流水并行策略（如 FThenB、1F1B、VPP 等）。本课题要求选手参与飞桨静态图流水并行相关能力建设，根据项目需求完成常见流水并行策略下相关代码的问题修复和功能增强，并对流水调度的性能进行针对性地分析和优化，提升自动并行架构的可用性。

#### 营员要求（1 人）：

- 熟悉 Python、C/C++
- 熟悉深度学习框架中的动态图和静态图
- 熟悉自动并行相关基础概念

### 项目五：动转静性能优化专项

#### 项目介绍：

Paddle 自 2.6 版本发布了新一代动转静技术 SOT（Symbolic Opcode Translator），通过自适应子图打断实现了极高成功率的动静转换，使用户能够只需要添加装饰器 `@to_static` 即可享受动转静带来的加速效果。但在实际模型场景中，SOT 仍然存在一些性能优化点，比如动态 shape 场景符号推导优化、NumPy 组网支持、异常支持、Guard 性能优化等。希望藉由本次护航计划，与社区同学一起探索 SOT 的性能优化，提升 SOT 的性能和稳定性，为编译器、自动并行等场景提供更加完备的转静基础设施。

#### 营员要求（2 人）：

- 精通 Python
- 熟悉 C/C++
- 了解深度学习框架动态图和静态图机制
- 熟悉 JIT 及相关编译技术（加分项）

### 项目六：PIR-TRT 算子 Converter 及单测开发

#### 项目介绍：

Paddle 推理在新的架构下集成了新的高性能推理引擎 PIR-TRT，使用该引擎的一个基本条件是需要能够将 paddle 算子映射为对应的 TRT 算子，当前依然有部分 Paddle 算子未开发对应的映射 Converter，为了能够支持全场景使用新的 PIR-TRT 推理引擎，需要将全部 Converter 开发完毕并且完成单测开发适配来保证 Converter 开发质量。

#### 营员要求（1 人）：

- 熟悉 Python
- 熟悉 PIR-TRT 基本执行情况，TensorRT 组网流程

### 项目七：大语言模型推理&服务化易用性提升专项

#### 项目介绍：

随着众多基于大模型重构的产品逐渐落地到我们的生活中，业界对大模型的推理部署的需求呈井喷式增长，2024 年是大模型推理方向飞速发展的一年，Paddle 支持大模型推理优化以来，迭代了一版又一版推理方案，推理性能逐渐走到了行业前列。但由于过去的野蛮生长、迭代，我们一定程度上忽视了推理框架的易用性、易开发性。本项目致力于提升大语言模型推理&服务化框架的易用性，具体的工作包括但不限于：

1. 全面清理历史旧代码；
2. 采用模块化组网方式全面升级组网方案；
3. 打通推理、服务化一键式部署流程。

#### 营员要求（1 人）：

- 熟悉 Python
- 熟悉流行大语言模型网络架构、了解大模型推理常用优化手段
- 熟悉 vLLM、TensorRT-LLM 等大模型推理框架实现优先
