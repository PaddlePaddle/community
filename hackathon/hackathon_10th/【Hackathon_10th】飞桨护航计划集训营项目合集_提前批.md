此文档展示 **PaddlePaddle Hackathon 第十期活动——飞桨护航计划集训营（提前批）**- 项目的详细介绍

## 赛题详情

### 项目一：PaddleOCR + ERNIE × Open-Source Ecosystem 高价值开源项目案例征集

#### 项目介绍：

我们正在推进 **PaddleOCR** 与 **ERNIE** 在产业级 AI 应用中的深度落地，并与开源生态头部项目联合打造高质量案例，包括：
* **PaddleOCR + ERNIE × Milvus**
* **PaddleOCR + ERNIE × Dify**
* **PaddleOCR + ERNIE × RAGFlow**
* **PaddleOCR + ERNIE × Pathway**
* **PaddleOCR + ERNIE × HayStack**
* **…**

我们正在寻找愿意一起 构建真正有行业价值的 AI 系统 的同学与开发者。详情可见：[PaddleOCR + ERNIE × Open-Source Ecosystem 高价值开源项目案例征集。](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/OCR%2BERNIE/PaddleOCR%20%2B%20ERNIE%20%C3%97%20Open-Source%20Ecosystem%20%E9%AB%98%E4%BB%B7%E5%80%BC%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E6%A1%88%E4%BE%8B%E5%BE%81%E9%9B%86.md)

#### 营员要求（2 人）：

- 具备基础 Python 能力
- 有简单的 AI/ML 项目经验更好
- 愿意深入学习、持续迭代
- 有项目思维，想做“能落地”的东西

### 项目二：Paddle API兼容性增强

#### 项目介绍：

为了降低新模型（特别是新的大模型）使用飞桨开发或迁移到飞桨的成本，飞桨从3.2版本开始了 **API兼容性适配** 工作，提升了API针对不同框架写法的自适应能力，**针对Pytorch项目，仅需修改代码前缀 torch为paddle，即可无缝迁移到Paddle。**
当前已完成了**700+**个重点API与Pytorch API的无缝兼容工作（包括：API名称、参数名称、参数个数、参数语义），具体名单详见 [PyTorch-Paddle API 官网映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html#id1)。但仍有很多Paddle API尚未完成兼容性适配 工作，此次工作旨在对Paddle API进行整体兼容性推全。

#### 营员要求（1-2 人）：

- 熟悉 Python开发，掌握C++开发
- 掌握CUDA开发更优
- 熟悉Paddle新增API的开发流程与技能

### 项目三：PaConvert转换工具建设

#### 项目介绍：

为了实现快速将PyTorch代码自动化转写为Paddle代码，提升模型迁移的效率，我们建设了[PaConvert代码自动转换工具](https://github.com/PaddlePaddle/PaConvert)：PaddlePaddle Code Convert Toolkits。
PaConvert搭建起Pytorch-Paddle之间的桥梁，也是《API兼容性增强》项目的验收标准，需协同《API兼容性增强》共同建设。转换工具目前支持1600+个Pytorch API的转换与兼容验证，所支持API均已在[Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)进行发布。针对转换工具与API映射表，目前仍有不少可继续完善地方，主要包括：[映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)自动生成工具、[映射文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/invok_only_diff/torch.Tensor.argwhere.html)内容校验工具、[映射文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/invok_only_diff/torch.Tensor.argwhere.html)错误修复、[Pytorch单测](https://github.com/PaddlePaddle/PaConvert/blob/master/tests/test_equal.py)增强、转换Bug修复、转换策略新增 等方面。

#### 营员要求（1 人）：

- 熟悉 Python开发
- 熟悉Pytorch框架的使用，熟练编写Pytorch API单测的各种用法
- 有论文复现与模型迁移经验更优

### 项目四：Paddle C++ API生态兼容建设

#### 项目介绍：

为加速生态融合、降低第三方高性能算子库（如deepgeem、flashinfer、flashMLA等）的迁移成本，PaddlePaddle框架推出了**C++ API兼容层**。我们在api/include/compat目录下提供了与PyTorch对齐的接口声明，旨在实现**编译时依赖切换即可完成迁移**，极大简化生态适配工作。
C++ API兼容层已进入积极开发阶段，现面向社区招募贡献者，共同推进以下工作：
* 持续扩展兼容性API接口覆盖
* 构建完备的单元测试体系

加入本项目，你将：
1. 深度理解PyTorch C++接口设计，并直接参与PaddlePaddle C++核心API的建设；
2. 系统掌握API从设计、实现到测试的全流程开发经验；
3. 熟练运用CMake构建跨平台C++项目，积累大型深度学习框架开发实战技能。

#### 营员要求（1-2 人）：

- 熟悉 C/C++
- 了解Pytorch/PaddlePaddle框架的使用
- 主动性强、善于沟通是加分项

### 项目五：面向大模型训练的高效分布式checkpoint系统研究

#### 项目介绍：

Checkpoint系统用于持久化保存模型参数、优化器、数据流、随机数等状态，以便训练任务发生故障时能够进行状态恢复。在模型预训练、后训练及推理等不同阶段衔接过程中，分布式策略和模型网络结构的实现差异会导致模型权重的分布状态发生变化，为此需要进行checkpoint的转换和迁移。业内常见做法是针对不同模型和任务定制转换脚本，这种方法开发和维护成本高，代码难以复用，极大地影响大模型训推研发效率。飞桨的分布式checkpoint系统基于统一的切分标记与参数重组描述，可灵活支持跨分布式策略和跨模型结构间的参数自动重切分，大幅降低权重转换成本。本课题学员将参与到飞桨分布式checkpoint系统的性能优化工作中，包括但不限于chcekpoint系统核心模块的代码开发和功能完善，分布式通信效率优化，文件系统读写性能优化等。

#### 营员要求（1 人）：

- 熟悉 Python和C++，有较强的编程和调试能力。
- 熟悉 NCCL通信库，有并行计算、分布式计算相关经验。
- 熟悉 Paddle、PyTorch和Megatron等训练框架。
- 熟悉操作系统及常见文件系统的设计原理。

### 项目六：GraphNet 计算子图构建与推广

#### 项目介绍：

任务一：GraphNet torch计算图反序列化可用性研究与优化。GraphNet项目会不断的序列化反序列化计算图。实践表明，torch计算图的反序列化很不稳定，形式化的来讲：torch_unserialize(torch_serialize(graph)) != graph。这个失败的概率大概在10%，而且torch不同版本成功率不一样。由于计算图反序列化是GraphNet子图切分工作的基石，我们有必要提升这里的可用性。

任务二：GraphNet torch计算图dtype泛化功能可用性研究与优化。GraphNet构建工具所抽取的计算图绝大部分都是fp32的参数类型，存在参数dtype代表性不足的问题。我们通过dtype泛化功能将fp32的计算图转为fp16/bf16，但此功能未打磨到位，亟待提升可用性。

任务三：编写ppt 《实践驱动的 AI 编译器技术问题：背景与分类》。当前AI编译器研究对于新手来说门槛太高。究其原因，新手对真实的问题背景了解不够。本项目希望罗列足够典型的编译器融合优化单测问题，以此未切入点阐明编译器怎样助力于AI模型优化。ppt内容描述：在 AI 模型大规模落地的过程中，跨硬件平台的部署面临严峻的性能、内存和功耗瓶颈。AI 编译器是解决这些挑战的关键技术。本主题以实践需求为驱动，深入剖析编译器在模型部署和优化中的技术问题背景。我们将聚焦图优化、数据流、算子融合、内存管理等核心领域，对当前亟待发力的具体优化难题进行系统化分类，为提升模型运行效率提供清晰的技术指引。

#### 营员要求（3 人）：

- GraphNet近期贡献者或对此有强烈兴趣。
- 有AI编译器开发背景。

### 项目七：FlashAttention 低精度训练算法研究与 Kernel 开发

#### 课题介绍：

在基于 Transformer 架构的大型语言模型（LLM）和多模态视觉语言模型（VLM）中，Attention 机制的计算开销随序列长度呈二次方增长，成为模型长文训练和推理的主要瓶颈。尽管 DeepSeek 已经端到端验证了 FP8 精度训练的可行性，但其优化主要集中在 Linear 层，而 Attention 算子的低精度训练仍有较大研究空间。
本项目旨在深入研究并实现支持 FP8/FP4 低精度的 Attention 算法，结合 Hopper 和 Blackwell 等最新 GPU 架构，针对硬件特性进行极致性能优化。项目将开发高效的 FlashAttention Kernel，系统性地验证低精度下 Attention 的收敛性与精度表现，推动大模型训练的高效化和低成本化，加速业界在大规模模型训练领域的创新步伐。

#### 营员要求（1 人）：

- 具备CUDA Kernel 优化经验
- 具备大模型训练的经验/论文发表经验
- 了解低精度计算
