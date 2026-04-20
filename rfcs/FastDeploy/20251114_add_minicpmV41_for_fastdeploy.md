# FastDeploy 新增 MiniCPM4.1-8B 模型

| 任务名称 | 【Hackathon 10th Spring No.50】为 FastDeploy 新增 MiniCPM4.1-8B 模型 |
|------|------|
| 提交作者 | bobby-cloudforge |
| 提交时间 | 2026-07-15 |
| 版本号 | V3.1 |
| 文件名 | 20251114_add_minicpmV41_for_fastdeploy.md |

# 一、概述

## 1、相关背景

MiniCPM4.1-8B 是 OpenBMB 发布的高性能紧凑型语言模型（8B 参数），采用 Grouped Query Attention (GQA) + μP (Maximal Update Parametrization) + LongRoPE 三大核心技术。该模型在 MMLU、HumanEval、GSM8K 等主流基准测试中表现优异，支持 65,536 token 长上下文窗口。

FastDeploy 作为飞桨高性能推理部署引擎，已有成熟的 GQA 注意力后端和量化框架。MiniCPM4.1-8B 的特殊之处在于 μP 三点缩放，需要在嵌入层、残差连接和 LM head 三处精确实现缩放因子。

## 2、功能目标

1. 实现 MiniCPM4.1-8B 模型组网代码，提交至 `FastDeploy/fastdeploy/model_executor/models/` 目录
2. 适配 FastDeploy 现有的低bit量化推理能力（WINT4/WINT8/FP8）
3. 提交模型使用说明文档
4. 支持 LongRoPE 位置编码（需适配 `rotary_embedding.py`）

## 3、意义

- 丰富 FastDeploy 可部署模型的覆盖范围，支持 OpenBMB MiniCPM 系列
- 为 8B 参数量级模型提供高效推理部署方案，适配消费级单卡 GPU 场景
- μP 缩放的正确实现为后续支持更多 μP 模型（如 CerebrasGPT）奠定基础

# 二、飞桨现状

## 1、已有基础设施

| 组件 | 文件 | 复用方式 |
|------|------|---------|
| GQA 注意力 | `layers/attention/attention.py` | 直接复用（`num_key_value_heads=2`） |
| QKV 并行线性层 | `layers/linear.py` (`QKVParallelLinear`) | 直接复用 |
| Merged 列并行 | `layers/linear.py` (`MergedColumnParallelLinear`) | 直接复用（gate+up 融合） |
| 行并行线性层 | `layers/linear.py` (`RowParallelLinear`) | 直接复用 |
| RMSNorm | `layers/normalization.py` | 直接复用 |
| 词表并行嵌入 | `layers/embeddings.py` (`VocabParallelEmbedding`) | 直接复用 |
| 并行 LM Head | `layers/lm_head.py` (`ParallelLMHead`) | 直接复用 |
| 旋转位置编码 | `layers/rotary_embedding.py` | 需扩展 LongRoPE 支持 |
| WINT 量化框架 | `layers/quantization/weight_only.py` | 直接复用 |
| 模型注册机制 | `models/__init__.py` (`ModelRegistry`) | 直接复用 |
| 权重映射 | `model_base.py` (`WeightsMapper`) | 直接复用 |

## 2、缺失组件

| 组件 | 说明 | 措施 |
|------|------|------|
| μP 缩放逻辑 | 嵌入/残差/LM head 三处缩放 | 在模型文件中实现 |
| LongRoPE 支持 | `rotary_embedding.py` 尚未处理 `longrope` 类型 | 扩展现有 RoPE 层 |
| 词表掩码 | `ori_vocab_size` 之外的 token 应被 mask 为 `-inf` | 在 `compute_logits` 中实现 |

# 三、业内方案调研

## 1、vLLM

vLLM 在 `vllm/model_executor/models/minicpm.py` 中实现了 MiniCPM 系列支持：
- 使用标准 GQA 注意力（非 MLA）
- μP 三点缩放参考实现：`scale_emb`（嵌入）、`scale_depth/√num_layers`（残差）、`1/(hidden_size/dim_model_base)`（lm_head）
- 注册为 `MiniCPMForCausalLM`

## 2、HuggingFace Transformers

`transformers/models/minicpm/modeling_minicpm.py` 提供原生实现：
- μP 缩放写入 `config.json`（`scale_emb=12`, `scale_depth=1.4`, `dim_model_base=256`）
- LongRoPE 通过 `rope_scaling` 配置项支持
- GQA: `num_attention_heads=32`, `num_key_value_heads=2`

## 3、对比分析

| 方案 | 优点 | 缺点 |
|------|------|------|
| vLLM | μP 实现清晰、张量并行完整 | Triton 内核对飞桨不适用 |
| HuggingFace | 原生配置加载 | 不支持 PaddlePaddle 后端 |
| **FastDeploy（本方案）** | 复用 GQA/量化框架、μP 精确移植 | 需新建模型文件 + LongRoPE 扩展 |

# 四、设计思路与实现方案

## 1、主体设计思路

采用 **Qwen2 + μP 扩展** 策略：模型结构与 Qwen2 高度同构（GQA + SiLU MLP + RMSNorm），在此基础上添加 μP 三点缩放。模型文件继承 `ModelForCasualLM`，通过 `@ModelRegistry.register_model_class()` 自动注册为 `MiniCPMForCausalLM`。

## 2、关键技术点

### 2.1 μP (Maximal Update Parametrization) 三点缩放

μP 确保模型在不同宽度间迁移时梯度和激活尺度保持稳定。MiniCPM4.1-8B 在三处应用缩放：

**Site 1 — 嵌入缩放 (×scale_emb)**
```python
# MiniCPM4Model.forward()
hidden_states = self.embed_tokens(input_ids)
hidden_states = hidden_states * self.config.scale_emb  # ×12
```
将嵌入向量放大 12 倍，补偿初始化时的缩小。

**Site 2 — 残差缩放 (×scale_depth/√num_hidden_layers)**
```python
# MiniCPM4DecoderLayer.forward()
scale = self.config.scale_depth / math.sqrt(self.config.num_hidden_layers)
# scale = 1.4 / √32 ≈ 0.2475

# 注意力残差
attn_output = self.self_attn(hidden_states, forward_meta)
hidden_states = residual + attn_output * scale

# MLP 残差
mlp_output = self.mlp(hidden_states, forward_meta)
hidden_states = residual + mlp_output * scale
```
每层的注意力和 MLP 输出均乘以 `scale_depth/√num_layers`，独立应用于两个子层。

**Site 3 — LM head 缩放 (÷ hidden_size/dim_model_base)**
```python
# MiniCPM4ForCausalLM.compute_logits()
lm_head_scale = self.config.hidden_size / self.config.dim_model_base
# = 4096 / 256 = 16.0

logits = hidden_states @ lm_head_weight.T
logits = logits / lm_head_scale  # ÷16
```
在 logit 计算后除以宽度比，防止大模型宽度下 logit 爆炸。

**关键验证**：三个缩放点的值和位置已与 [vLLM 参考实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/minicpm.py) 逐行对比确认。

### 2.2 GQA 注意力

MiniCPM4.1-8B 使用标准 Grouped Query Attention：
- `num_attention_heads = 32`（Q heads）
- `num_key_value_heads = 2`（KV heads，每个服务 16 个 Q head）
- `head_dim = hidden_size / num_attention_heads = 128`

直接复用 FastDeploy 的 `QKVParallelLinear` + `Attention` 层，无需自定义注意力后端：

```python
class MiniCPM4Attention(nn.Layer):
    def __init__(self, fd_config, layer_id, prefix=""):
        self.qkv_proj = QKVParallelLinear(fd_config, prefix=f"{prefix}.qkv_proj", with_bias=False)
        self.o_proj = RowParallelLinear(fd_config, prefix=f"{prefix}.o_proj", ...)
        self.attn = Attention(fd_config, layer_id=layer_id, prefix=prefix)
```

### 2.3 LongRoPE 位置编码

MiniCPM4.1-8B 使用 `longrope` 类型的位置编码，通过 `long_factor` 和 `short_factor` 数组按维度缩放频率：

```json
"rope_scaling": {
    "rope_type": "longrope",
    "long_factor": [0.998, 1.033, ..., 63.009],
    "short_factor": [0.998, 1.033, ..., 63.009],
    "original_max_position_embeddings": 65536
}
```

需在 `rotary_embedding.py` 中新增 `longrope` 分支，将 factor 数组乘入 inverse frequency 向量。

### 2.4 词表掩码

MiniCPM4.1-8B 的 `vocab_size=73448`，模型配置含 `ori_vocab_size`（原始词表大小）。在 `compute_logits` 中对超出 `ori_vocab_size` 的 token 应用 `-inf` 掩码，防止生成 padding token：

```python
logits[:, self.config.ori_vocab_size:] = float("-inf")
```

### 2.5 模型结构

```
MiniCPM4ForCausalLM (registered as "MiniCPMForCausalLM")
├── minicpm4: MiniCPM4Model
│   ├── embed_tokens: VocabParallelEmbedding (vocab=73448, dim=4096)
│   ├── layers[0..31]: MiniCPM4DecoderLayer ×32
│   │   ├── self_attn: MiniCPM4Attention
│   │   │   ├── qkv_proj: QKVParallelLinear (no bias, GQA 32:2)
│   │   │   ├── o_proj: RowParallelLinear
│   │   │   └── attn: Attention (neox RoPE)
│   │   ├── mlp: MiniCPM4MLP
│   │   │   ├── up_gate_proj: MergedColumnParallelLinear (SiLU, 16384×2)
│   │   │   └── down_proj: RowParallelLinear
│   │   ├── input_layernorm: RMSNorm
│   │   └── post_attention_layernorm: RMSNorm
│   └── norm: RMSNorm (final)
└── lm_head: ParallelLMHead (tie_word_embeddings=False)
```

### 2.6 权重映射

从 HuggingFace `openbmb/MiniCPM4.1-8B` 到 FastDeploy 的参数名映射：

| HuggingFace | FastDeploy | 变换 |
|-------------|-----------|------|
| `model.layers.{i}.self_attn.q_proj` | `minicpm4.layers.{i}.self_attn.qkv_proj` | QKV 堆叠 (shard: `q`) |
| `model.layers.{i}.self_attn.k_proj` | `minicpm4.layers.{i}.self_attn.qkv_proj` | QKV 堆叠 (shard: `k`) |
| `model.layers.{i}.self_attn.v_proj` | `minicpm4.layers.{i}.self_attn.qkv_proj` | QKV 堆叠 (shard: `v`) |
| `model.layers.{i}.mlp.gate_proj` | `minicpm4.layers.{i}.mlp.up_gate_proj` | gate+up 融合 (shard: 0) |
| `model.layers.{i}.mlp.up_proj` | `minicpm4.layers.{i}.mlp.up_gate_proj` | gate+up 融合 (shard: 1) |
| `model.` (前缀) | `minicpm4.` (前缀) | 前缀替换 |

直通加载（无变换）：`o_proj`, `down_proj`, `input_layernorm`, `post_attention_layernorm`, `norm`, `embed_tokens`, `lm_head`

使用 `WeightsMapper` 实现前缀重命名，`stacked_params_mapping` 实现 QKV 和 gate/up 融合。

### 2.7 张量并行

| 参数 | 切分方式 | 说明 |
|------|---------|------|
| `qkv_proj` | Column split | Q/K/V 按 head 切分 |
| `o_proj` | Row split | 输出拼接 |
| `up_gate_proj` | Column split | gate 和 up 各自按列切分 |
| `down_proj` | Row split | 输出拼接 |
| `embed_tokens` | Vocab split | 按词表切分 |
| `lm_head` | Vocab split | 按词表切分 |

### 2.8 文件结构

| 路径 | 说明 | 行数 |
|------|------|------|
| `fastdeploy/model_executor/models/minicpm4.py` | 模型组网代码 | ~516 |
| `tests/model_executor/test_minicpm4.py` | 24 个 pytest 单元测试 | ~514 |
| `docs/best_practices/MiniCPM4-8B.md` | 使用说明文档 | ~104 |
| `docs/supported_models.md` | 更新支持模型列表 | +1 |

# 五、测试和验收的考量

## 1、单元测试（CPU，无 GPU 依赖）
- 24 个 pytest 测试覆盖 MLP/Attention/DecoderLayer/Model/CausalLM 各层
- 使用 `monkeypatch.setattr` 替换重基础设施（注意力后端、并行线性层等），8 个 stub 类
- 验证 μP 三点缩放值（×12 嵌入、÷16 lm_head、残差缩放）
- 验证权重映射、QKV 堆叠、gate/up 融合、张量并行切分
- **结果**：24/24 PASSED，2.16 秒（A800-SXM4-80GB）

## 2、集成测试（GPU，真实模型权重）
- 6 个端到端测试：启动 FastDeploy API Server + 真实推理
- 验证：健康检查、模型列表、算术推理准确性、无 `<unk>` 泄漏、多轮对话
- **环境**：NVIDIA A800-SXM4-80GB

## 3、量化精度
- WINT8/WINT4/FP8 通过标准 FastDeploy 层接入，与 Qwen2/DeepSeek 同源

# 六、影响面

## 对用户的影响
- 新增模型选项，无 breaking changes
- 通过标准 FastDeploy CLI 启动即可使用：`python -m fastdeploy.entrypoints.openai.api_server --model openbmb/MiniCPM4.1-8B`

## 对框架架构的影响
- 新增一个模型文件 `minicpm4.py`，不修改现有模型代码
- `rotary_embedding.py` 需新增 LongRoPE 分支（~20 行，向后兼容）

## 对性能的影响
- 不影响现有模型性能
- μP 缩放为纯乘法运算，开销可忽略

# 七、排期规划

| 阶段 | 任务 | 状态 |
|------|------|------|
| Phase 0 | 环境搭建 + 模型配置分析 + vLLM 参考研究 | ✅ 已完成 |
| Phase 1 | 模型组网（GQA + μP + 权重加载）| ✅ 已完成（516 行） |
| Phase 2 | 单元测试 | ✅ 已完成（24 tests, 24/24 PASSED） |
| Phase 3 | 使用说明文档 | ✅ 已完成 |
| Phase 4 | 集成测试（GPU + 真实权重） | 进行中 |
| Phase 5 | Code Review + 合入 | 待提交 |

# 附件及参考资料

1. MiniCPM4.1-8B HuggingFace: https://huggingface.co/openbmb/MiniCPM4.1-8B
2. MiniCPM 技术报告: https://arxiv.org/abs/2404.06395
3. FastDeploy Qwen2 模型（结构参考）: `fastdeploy/model_executor/models/qwen2.py`
4. vLLM MiniCPM 实现: `vllm/model_executor/models/minicpm.py`
5. μP 论文 (Yang et al., 2022): https://arxiv.org/abs/2203.03466
6. FastDeploy 代码 PR: https://github.com/PaddlePaddle/FastDeploy/pull/7332

<!-- RFC v3.1 — MiniCPM4.1-8B model support -->
