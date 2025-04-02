# 在 PaddleNLP 中复现 ModernBert 模型

|任务名称 | 在 PaddleNLP 中复现 ModernBert 模型 | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | robinbg | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2025-04-02 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20250402_add_modernbert_for_paddlenlp.md<br> |


# 一、概述
## 1、相关背景

ModernBert 是一种现代优化的 BERT 变体，在 2 万亿个词元上训练，支持 8192 长度的序列。相比原始 BERT，ModernBert 在性能与模型大小之间提供了更好的权衡，主要通过以下改进：

- 使用旋转位置编码 (Rotary Positional Embeddings)，替代传统的绝对位置编码
- 采用 GeGLU 激活函数，替代传统的 GELU
- 实现滑动窗口注意力机制 (Sliding Window Attention)，有效处理长序列
- 针对推理速度优化的架构设计

这些改进使得 ModernBert 在相同参数量下具有更强的性能，且更适合处理长序列输入。

## 2、功能目标

在 PaddleNLP 中实现 ModernBert 模型，对齐 HuggingFace transformers 中的实现，确保在 fp32 下误差小于 1e-5，bf16 下误差小于 1e-3。

## 3、意义

ModernBert 模型的引入将增强 PaddleNLP 处理长文本的能力，为用户提供更高效的预训练模型选择，同时保持与 HuggingFace 生态的兼容性。

# 二、飞桨现状

PaddleNLP 目前已支持多种 Transformer 架构模型，包括 BERT、RoBERTa、ERNIE 等，但尚未支持具有旋转位置编码、GeGLU 激活函数和优化的长序列处理能力的 ModernBert 模型。现有的 BERT 实现可以作为实现 ModernBert 的基础，但需要进行架构上的修改以适配 ModernBert 的特性。

# 三、业内方案调研

目前，HuggingFace transformers 库已实现 ModernBert 模型，地址为：
https://github.com/huggingface/transformers/tree/main/src/transformers/models/modernbert

该实现包含了完整的模型架构、预训练权重以及相关的文档，是 PaddleNLP 实现 ModernBert 的主要参考。ModernBert 模型在 HuggingFace 上已有多个开源检查点，并展示了其在多种下游任务上的出色表现。

# 四、对比分析

与 PaddleNLP 现有的 BERT 实现相比，ModernBert 需要做以下关键改进：

1. **位置编码**：从静态绝对位置编码转变为旋转位置编码 (RoPE)
2. **激活函数**：从 GELU 转变为 GeGLU
3. **注意力机制**：实现滑动窗口注意力，支持处理长序列
4. **模型配置**：添加新的配置参数以支持 ModernBert 特有的功能

这些改进将使 ModernBert 能够处理更长的序列（最大 8192 个词元），同时提供更好的性能-大小比。

# 五、设计思路与实现方案

ModernBert 模型在 PaddleNLP 中的实现将参考现有的 BERT 实现结构，主要包含以下几个关键文件：

1. **configuration.py**：定义 ModernBertConfig 类，包含模型所需的配置参数
2. **modeling.py**：实现模型的核心架构
3. **tokenizer.py**：与 BERT tokenizer 兼容，复用现有实现

## 1. ModernBertConfig 配置类

在 `paddlenlp/transformers/modernbert/configuration.py` 中，需要定义 ModernBertConfig 类，继承自 PretrainedConfig：

```python
from paddlenlp.transformers.configuration_utils import PretrainedConfig

class ModernBertConfig(PretrainedConfig):
    """
    ModernBert 模型的配置类，包含所有架构参数
    """
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="geglu",  # 使用 GeGLU 替代 GELU
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=8192,  # 支持更长的序列
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        sliding_window_size=512,  # 滑动窗口大小
        tie_word_embeddings=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.sliding_window_size = sliding_window_size
        self.tie_word_embeddings = tie_word_embeddings
```

## 2. 实现旋转位置编码 (RoPE)

在 `paddlenlp/transformers/modernbert/modeling.py` 中，需要实现旋转位置编码：

```python
def apply_rotary_position_embeddings(q, k, cos, sin, position_ids):
    # 实现旋转位置编码的逻辑
    # 参考 HuggingFace transformers 中 ModernBert 的实现
```

## 3. 实现 GeGLU 激活函数

在 `paddlenlp/transformers/modernbert/modeling.py` 中，实现 GeGLU 激活函数：

```python
def geglu(x, gate):
    # 实现 GeGLU 激活函数
    # GeGLU(x, gate) = x * GELU(gate)
```

## 4. 实现滑动窗口注意力

在注意力机制中实现滑动窗口注意力，以高效处理长序列：

```python
def sliding_window_attention(query, key, value, window_size, attention_mask=None):
    # 实现滑动窗口注意力
    # 限制每个 token 只能与其左右 window_size 范围内的 token 进行注意力计算
```

# 六、测试验收的考量

ModernBert 模型的实现需要通过以下测试验证：

1. **精度对齐测试**：
   - 使用 [PaDiff](https://github.com/PaddlePaddle/PaDiff) 工具验证与 PyTorch 实现的对齐程度
   - 在 fp32 下误差需小于 1e-5
   - 在 bf16 下误差需小于 1e-3

2. **单元测试**：
   - 为 ModernBert 添加单元测试，验证各个组件的功能正确性
   - 测试 RoPE、GeGLU、滑动窗口注意力等关键功能

3. **功能测试**：
   - 验证模型能够正确处理不同长度的输入序列
   - 验证模型在下游任务（如文本分类、序列标注等）上的性能

4. **兼容性测试**：
   - 验证与 HuggingFace 预训练权重的兼容性
   - 验证与 PaddleNLP 生态的兼容性

# 七、可行性分析和排期规划

基于现有的 BERT 实现和 HuggingFace 的 ModernBert 实现，在 PaddleNLP 中复现 ModernBert 是可行的。实现计划如下：

1. **第一阶段（1周）**：
   - 分析 HuggingFace ModernBert 代码结构
   - 设计 PaddleNLP 中的 ModernBert 实现方案
   - 实现 ModernBertConfig 类

2. **第二阶段（2周）**：
   - 实现 ModernBert 核心组件：RoPE、GeGLU、滑动窗口注意力
   - 实现 ModernBertModel 类及相关的下游任务模型类

3. **第三阶段（1周）**：
   - 实现与 HuggingFace 权重兼容的转换函数
   - 进行精度对齐验证

4. **第四阶段（1周）**：
   - 编写单元测试和文档
   - 修复问题和优化性能

# 八、影响面

ModernBert 的实现将对 PaddleNLP 产生以下影响：

1. 增强 PaddleNLP 处理长文本的能力，支持最大 8192 长度的序列
2. 提供更高效的预训练模型选择，在相同参数量下提供更好的性能
3. 完善 PaddleNLP 的模型生态，与业界最新进展保持同步
4. 为后续实现更多基于旋转位置编码和 GeGLU 的模型奠定基础

# 名词解释

- **RoPE (Rotary Position Embedding)**: 旋转位置编码，一种能更好处理相对位置信息的位置编码方法
- **GeGLU**: Gated GELU 激活函数，相比传统的 GELU 提供更好的性能
- **滑动窗口注意力**: 一种注意力计算优化方式，限制每个 token 只关注局部上下文，提高长序列处理效率

# 附件及参考资料

1. [ModernBert 论文](https://arxiv.org/abs/2412.13663)
2. [HuggingFace ModernBert 实现](https://github.com/huggingface/transformers/tree/main/src/transformers/models/modernbert)
3. [PaddleNLP 模型对齐指南](https://paddlenlp.readthedocs.io/zh/latest/community/contribute_models/align_pytorch_and_paddle.html)
