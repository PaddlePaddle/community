# FastDeploy 新增 MiniCPM4.1-8B 模型

| 任务名称 | 【Hackathon 10th Spring No.50】为 FastDeploy 新增 MiniCPM4.1-8B 模型 |
|------|------|
| 提交作者 | cloudforge1 |
| 提交时间 | 2026-07-08 |
| 版本号 | V1.0 |
| 文件名 | 20260708_add_minicpm41_8b_for_fastdeploy.md |

# 一、概述

## 1、相关背景

MiniCPM4.1-8B 是 OpenBMB 发布的高性能轻量级语言模型，参数量仅 8B，但通过 Multi-head Latent Attention (MLA)、InfLLM-V2 稀疏注意力和 DoubleSparse 等技术实现了超越同规模模型的推理性能。该模型在长文本理解、代码生成和多轮对话等场景表现优异。

FastDeploy 作为飞桨高性能推理部署引擎，已通过 DeepSeek-v3 的实现积累了成熟的 MLA 注意力后端基础设施（`mla_attention_backend.py`），为 MiniCPM4.1-8B 的接入提供了天然的组件复用能力。

## 2、功能目标

1. 实现 MiniCPM4.1-8B 模型组网代码，提交至 `FastDeploy/fastdeploy/model_executor/models/` 目录
2. 适配 FastDeploy 现有的低bit量化推理能力（WINT2/WINT4/WINT8/FP8）
3. 提交模型使用说明文档
4. 如需开发自定义算子，提交至 `FastDeploy/custom_ops/gpu_ops/` 目录

## 3、意义

- 丰富 FastDeploy 可部署模型的覆盖范围，支持 OpenBMB MiniCPM 系列
- 为 8B 参数量级模型提供高效推理部署方案，适配消费级 GPU 场景
- 复用已有 MLA 基础设施，验证 FastDeploy 注意力后端的通用性

# 二、飞桨现状

## 1、已有基础设施

FastDeploy 已通过 DeepSeek-v3/v3.2 的实现建立了丰富的可复用组件：

| 组件 | 文件 | 复用程度 |
|------|------|---------|
| MLA 注意力后端 | `layers/attention/mla_attention_backend.py` | 直接复用 |
| RMSNorm | `layers/normalization.py` | 直接复用 |
| 并行线性层 | `layers/linear.py` (Column/Row/Merged) | 直接复用 |
| 旋转位置编码 | `layers/rotary_embedding.py` | 直接复用 |
| WINT 量化框架 | `layers/quantization/weight_only.py` | 直接复用 |
| 模型注册机制 | `models/__init__.py` (auto_models_registry) | 直接复用 |
| 权重加载 | `model_base.py` + `PretrainedModel` | 直接复用 |

## 2、缺失组件

| 组件 | 说明 | 复杂度 |
|------|------|--------|
| InfLLM-V2 稀疏注意力 | 长上下文 token 分块检索 + 稀疏计算 | 高 — 标记为未来工作 |
| DoubleSparse 注意力 | 动态稀疏模式选择 | 中 — 标记为未来工作 |
| MiniCPM4.1 模型文件 | 尚未创建 | 低 — 本 RFC 核心目标 |

# 三、业内方案调研

## 1、vLLM

vLLM 已支持 MiniCPM 系列模型，采用 HuggingFace-compatible 的配置加载方式。vLLM 的实现分为两部分：基础模型（MLA + dense layers）和高级特性（InfLLM-V2 等）。基础模型部分直接复用现有的 GQA/MLA 注意力后端。

## 2、TGI (Text Generation Inference)

HuggingFace TGI 通过 `transformers` 库原生支持 MiniCPM4.1-8B，使用标准 HuggingFace `AutoModelForCausalLM` 加载路径。

## 3、对比分析

| 方案 | 优点 | 缺点 |
|------|------|------|
| vLLM | MLA 注意力优化、张量并行 | Triton 内核对飞桨不适用 |
| TGI | 原生 HuggingFace 兼容 | 不支持 PaddlePaddle 后端 |
| **FastDeploy（本方案）** | 复用已有 MLA 后端、全量化支持 | 需新建模型文件 |

# 四、设计思路与实现方案

## 1、主体设计思路

采用**最大化复用**策略：将 MiniCPM4.1-8B 的 MLA 注意力映射到 FastDeploy 已有的 `MLAAttentionBackend`（源自 DeepSeek-v3），dense transformer 层复用 `qwen3.py` 的 MLP 模式。模型文件继承 `ModelForCasualLM`，通过 `@ModelRegistry.register_model_class()` 自动注册。

**分阶段交付**：
- Phase A（核心）：基础 MiniCPM4.1-8B 模型 + MLA + 低bit量化 — 本 RFC 范围
- Phase B（未来工作）：InfLLM-V2 稀疏注意力后端 — 后续追加

## 2、关键技术点

### 2.1 MLA 注意力适配

MiniCPM4.1-8B 使用 MLA（Multi-head Latent Attention），将 KV 投影压缩到低秩潜在空间：

```
KV cache shape: [max_blocks, 1, block_size, kv_lora_rank + qk_rope_head_dim]
```

FastDeploy 已有 `MLAAttentionBackend` 支持此模式，关键参数映射：

| MiniCPM4.1 配置 | FD MLA 参数 | 值 |
|-----------------|------------|-----|
| `kv_lora_rank` | `kv_lora_rank` | 512 |
| `qk_nope_head_dim` | `qk_nope_head_dim` | 128 |
| `qk_rope_head_dim` | `qk_rope_head_dim` | 64 |
| `num_attention_heads` | `num_heads` | 64 |
| `num_key_value_heads` | `kv_num_heads` | 1 (压缩后) |

### 2.2 模型结构

```python
@ModelRegistry.register_model_class("MiniCPM41ForCausalLM", ["minicpm4.1"])
class MiniCPM41ForCausalLM(ModelForCasualLM):
    """MiniCPM4.1-8B 推理模型"""
    
    class MiniCPM41DecoderLayer:
        """单层 decoder: MLA attention + SwiGLU MLP + RMSNorm"""
        - self_attn: MiniCPM41Attention (基于 MLAAttentionBackend)
        - mlp: MiniCPM41MLP (SwiGLU，复用 ColumnParallelLinear)
        - input_layernorm: RMSNorm
        - post_attention_layernorm: RMSNorm
    
    class MiniCPM41Model:
        - embed_tokens: VocabParallelEmbedding
        - layers: List[MiniCPM41DecoderLayer] × 40
        - norm: RMSNorm
    
    class MiniCPM41ForCausalLM:
        - model: MiniCPM41Model
        - lm_head: ParallelLMHead
```

### 2.3 文件结构

| 路径 | 说明 |
|------|------|
| `fastdeploy/model_executor/models/minicpm4.py` | 模型组网代码（~400-500行） |
| `docs/zh/supported_models.md` | 更新支持模型列表 |
| `tests/models/test_minicpm4.py` | 基本加载和推理测试 |

### 2.4 权重映射

从 HuggingFace `openbmb/MiniCPM4.1-8B` 到 FastDeploy 的参数名映射：

| HuggingFace | FastDeploy |
|-------------|-----------|
| `model.layers.{i}.self_attn.q_a_proj` | `model.layers.{i}.self_attn.q_a_proj` |
| `model.layers.{i}.self_attn.kv_a_proj_with_mqa` | `model.layers.{i}.self_attn.kv_a_proj` |
| `model.layers.{i}.self_attn.q_b_proj` | `model.layers.{i}.self_attn.q_b_proj` |
| `model.layers.{i}.self_attn.kv_b_proj` | `model.layers.{i}.self_attn.kv_b_proj` |
| `model.layers.{i}.mlp.gate_proj` | `model.layers.{i}.mlp.gate_up_proj` (merged) |
| `model.layers.{i}.mlp.up_proj` | `model.layers.{i}.mlp.gate_up_proj` (merged) |
| `model.layers.{i}.mlp.down_proj` | `model.layers.{i}.mlp.down_proj` |

### 2.5 量化支持

直接复用 FastDeploy 现有量化框架：
- **WINT8/WINT4/WINT2**: 通过 `weight_only_linear` 算子，已验证于 DeepSeek/Qwen 系列
- **FP8**: 通过 FP8 线性层实现

# 五、测试和验收的考量

1. **模型加载测试**: 验证从 HuggingFace 权重正确加载到 FastDeploy
2. **推理正确性**: 对比 HuggingFace transformers 的输出（token-level 一致性）
3. **量化精度**: WINT8 精度损失 <1%，WINT4 精度损失 <3%
4. **性能基线**: 单卡 A100 首 token 延迟和生成吞吐量
5. **兼容性**: 支持 `python -m fastdeploy.entrypoints.openai.api_server --model openbmb/MiniCPM4.1-8B`

# 六、影响面

## 对用户的影响
- 新增模型选项，无 breaking changes
- 通过标准 FastDeploy CLI 启动即可使用

## 对框架架构的影响
- 新增一个模型文件 `minicpm4.py`，不修改现有代码
- 复用 `MLAAttentionBackend`，不新增后端

## 对性能的影响
- 不影响现有模型性能
- MLA 压缩 KV 缓存可降低内存消耗

# 七、排期规划

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| Phase 0 | 环境搭建 + 权重下载 + 配置解析 | 1天 |
| Phase 1 | 模型组网（MLA + MLP + 权重加载） | 3-4天 |
| Phase 2 | 量化集成（WINT8/WINT4/WINT2/FP8） | 2天 |
| Phase 3 | 测试 + 文档 + Code Review | 2-3天 |
| **合计** | | **~1.5周** |

# 附件及参考资料

1. MiniCPM4.1-8B HuggingFace: https://huggingface.co/openbmb/MiniCPM4.1-8B
2. MiniCPM 技术报告: https://arxiv.org/abs/2404.06395
3. FastDeploy MLA 后端: `fastdeploy/model_executor/layers/attention/mla_attention_backend.py`
4. FastDeploy DeepSeek-v3 模型（MLA 参考）: `fastdeploy/model_executor/models/deepseek_v3.py`
5. 已合并的 H9 MiniCPM4.1 RFC: https://github.com/PaddlePaddle/community/pull/1183
6. vLLM MiniCPM 支持: https://github.com/vllm-project/vllm
