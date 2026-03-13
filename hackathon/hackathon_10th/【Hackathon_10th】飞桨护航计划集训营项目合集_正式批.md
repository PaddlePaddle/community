# PaddlePaddle Hackathon 第十期活动——飞桨护航计划集训营（正式批）项目详情

此文档展示 **PaddlePaddle Hackathon 第十期活动——飞桨护航计划集训营（正式批）** 项目的详细介绍

## 赛题详情

### 项目一：基于AI Coding的飞桨-Torch编译能力融合探索

**项目介绍：**
随着深度学习框架的演进，编译优化已成为提升模型性能的关键。本项目立足于飞桨（PaddlePaddle）积极拥抱开源生态的优势，旨在探索将PyTorch的`torch.compile`（动态图编译）能力集成至飞桨框架中。

不同于传统的人工硬编码，本项目将**主要通过AI Coding（AI辅助编程）**开展核心开发。利用大模型强大的代码生成与转换能力，我们将尝试自动适配PyTorch的编译优化功能，以此验证"AI生成编译优化代码"在深度学习框架底层的可行性。

这不仅是一次跨框架技术融合的尝试，更是对"AI for AI"开发范式的实践。通过引入新的编译技术，作为飞桨现有编译优化体系的能力补充，探索更精细的图捕获与算子融合策略，进一步拓宽飞桨的性能优化上限。

**营员要求（1人）：**
- 有 Vibe Coding 开发经验
- 熟悉深度学习框架中的动态图和静态图，对 torch 的 compile 原理有深入了解

---

### 项目二：PaddleOCR + ERNIE × Open-Source Ecosystem 高价值开源项目案例征集

**项目介绍：**
我们正在推进 **PaddleOCR** 与 **ERNIE** 在产业级 AI 应用中的深度落地，并与开源生态头部项目联合打造高质量案例，包括：
- PaddleOCR + ERNIE × Milvus
- PaddleOCR + ERNIE × Dify
- PaddleOCR + ERNIE × RAGFlow
- PaddleOCR + ERNIE × Pathway
- PaddleOCR + ERNIE × HayStack
- …

详情可见：[PaddleOCR + ERNIE × Open-Source Ecosystem 高价值开源项目案例征集](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/OCR%2BERNIE/PaddleOCR%20%2B%20ERNIE%20%C3%97%20Open-Source%20Ecosystem%20%E9%AB%98%E4%BB%B7%E5%80%BC%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E6%A1%88%E4%BE%8B%E5%BE%81%E9%9B%86.md)。

**营员要求（2人）：**
- 具备基础 Python 能力
- 有简单的 AI/ML 项目经验更好
- 愿意深入学习、持续迭代
- 有项目思维，想做"能落地"的东西

---

### 项目三：Paddle API兼容性增强

**项目介绍：**
为了降低新模型（特别是新的大模型）使用飞桨开发或迁移到飞桨的成本，飞桨从3.2版本开始了 **API兼容性适配** 工作，提升了API针对不同框架写法的自适应能力，**针对Pytorch项目，仅需修改代码前缀 `torch` 为 `paddle`，即可无缝迁移到Paddle**。

当前已完成了**900+个**API与Pytorch API的无缝兼容工作（包括：API名称、参数名称、参数个数、参数语义），具体名单详见 [PyTorch-Paddle API 官网映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html#id1)。但仍有很多Paddle API尚未完成兼容性适配工作，此次工作旨在对Paddle API进行整体兼容性推全。包括API机制增强、API功能开发等各项API/OP建设工作。

**营员要求（2人）：**
- 熟悉 Python开发，掌握C++开发
- 掌握CUDA开发更优
- 熟悉Paddle新增API的开发流程与技能

---

### 项目四：PaConvert转换工具建设+官网映射文档建设

**项目介绍：**
为了实现快速将PyTorch代码自动化转写为Paddle代码，提升模型迁移的效率，我们建设了[PaConvert代码自动转换工具](https://github.com/PaddlePaddle/PaConvert)：**Pa**ddlePaddle Code **Convert** Toolkits。

PaConvert搭建起Pytorch-Paddle之间的桥梁，也是《API兼容性增强》项目的验收标准，需协同《API兼容性增强》共同建设。转换工具目前支持1600+个Pytorch API的转换与兼容验证，所支持API均已在[Pytorch-Paddle API映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)进行发布。针对转换工具与API映射表，目前仍有不少可继续完善的地方，主要包括：[官网映射表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/pytorch_api_mapping_cn.html)自动生成工具、[映射文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/model_convert/convert_from_pytorch/api_difference/invok_only_diff/torch.Tensor.argwhere.html)内容校验工具、API差异文档错误修复、API文档错误修复、Pytorch单测增强、转换Bug修复等方面。

**营员要求（1人）：**
- 熟悉 Python开发
- 熟悉Pytorch框架的使用，熟练编写Pytorch API单测的各种用法
- 有论文复现与模型迁移经验更优

---

### 项目五：面向大模型训练的高效分布式checkpoint系统研究

**项目介绍：**
Checkpoint系统用于持久化保存模型参数、优化器、数据流、随机数等状态，以便训练任务发生故障时能够进行状态恢复。在模型预训练、后训练及推理等不同阶段衔接过程中，分布式策略和模型网络结构的实现差异会导致模型权重的分布状态发生变化，为此需要进行checkpoint的转换和迁移。业内常见做法是针对不同模型和任务定制转换脚本，这种方法开发和维护成本高，代码难以复用，极大地影响大模型训推研发效率。飞桨的分布式checkpoint系统基于统一的切分标记与参数重组描述，可灵活支持跨分布式策略和跨模型结构间的参数自动重切分，大幅降低权重转换成本。本课题学员将参与到飞桨分布式checkpoint系统的性能优化工作中，包括但不限于checkpoint系统核心模块的代码开发和功能完善，分布式通信效率优化，文件系统读写性能优化等。

**营员要求（1人）：**
- 熟悉 Python和C++，有较强的编程和调试能力
- 熟悉 NCCL通信库，有并行计算、分布式计算相关经验
- 熟悉 Paddle、PyTorch和Megatron等训练框架
- 熟悉操作系统及常见文件系统的设计原理

---

### 项目六：大模型低资源训练能力与优化策略研究

**项目介绍：**
随着大模型参数规模的持续增长，其全量预训练与微调过程对计算资源、存储资源和时间成本提出了极高的要求，极大地限制了广大研究机构与企业在资源受限环境下的技术创新与应用落地。如何在有限的GPU内存、计算卡数和训练时长等约束下，高效地完成大模型的训练与适配，已成为当前产业界与学术界关注的焦点。本课题旨在系统性地研究并整合前沿的低资源训练与优化技术，构建一套智能、高效的分布式训练加速与资源节省系统，通过算法与工程的协同优化，在有限硬件资源下实现大模型的高效训练。

**营员要求（1人）：**
- 熟悉Python和C++，具备扎实的编程能力和系统调试经验
- 熟悉深度学习框架（如PaddlePaddle, PyTorch）的基本原理和机制，有模型训练或调优经验者优先
- 对大模型相关技术有浓厚兴趣，了解参数高效微调、模型量化、混合精度训练、分布式训练等其中至少一项技术
- 具备强烈的责任心、良好的团队协作能力和主动学习精神

---

### 项目七：PaddlePaddle 基础组件升级

**项目介绍：**
作为深耕高性能计算的深度学习框架，底层的工具链和语言标准直接决定了框架的开发效率、安全性及运行时性能。为了紧跟开源社区前沿并利用最新的编译器优化手段，本项目旨在推动 PaddlePaddle 核心仓库及其开发工具链的全面现代化升级，确保在多平台环境下保持技术领先性。

主要任务包括将 Python 环境适配至 **Python 3.14**，将 Linux 编译器基准升级至 **GCC 15**，并将底层 C++ 语言标准从现有版本全面提升至 **C++20**。升级过程涉及处理新标准下的语法兼容性冲突、优化代码以利用 C++20 特性（如 Concepts、Modules 或性能更优的 STL 实现），并解决 Python 新版本带来的 API 变更。同时，本项目需**同步跟进 Windows 平台 Visual Studio 2022 的编译器更新**，确保 C++20 新特性在 MSVC 环境下的稳定支持与性能对齐。此外，需同步更新 **CI（持续集成）系统的全套 Docker 镜像及 Windows 构建环境**，重构构建脚本，确保在全平台全新的工具链环境下，PaddlePaddle 的编译、单元测试及分布式集成测试能够稳健运行。

**营员要求（1人）：**
- 熟悉 **Python** 与 **C/C++**，具有底层开发或大型项目迁移经验
- 熟悉 **CMake** 编译体系，了解 Linux（GCC）与 **Windows（MSVC/VS2022）** 环境下的工具链配置与调试
- 熟悉 **Docker** 容器技术及大型开源项目的 **CI/CD** 流程
- 对 **C++20** 新特性（如 Ranges, Coroutines, Concepts）有深入了解，并具备处理复杂语法兼容性问题能力优先
