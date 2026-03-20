# FastDeploy 新增 MiniMax-M1 模型

| 任务名称 | 【Hackathon 10th Spring No.47】为 FastDeploy 新增 MiniMax-M1 模型 |
|------|------|
| 提交作者 | cloudforge1 |
| 提交时间 | 2026-07-08 |
| 版本号 | V1.0 |
| 文件名 | 20260708_add_minimax_m1_for_fastdeploy.md |

# 一、概述

## 1、相关背景

MiniMax-M1 是 MiniMax 发布的 4560 亿参数混合注意力大语言模型，采用 80 层混合架构：标准 GQA（全注意力层）和线性注意力/Mamba 层交替排列，结合 MoE（Mixture of Experts）+ SharedExpert 和 DeepNorm 预归一化策略。该模型在长上下文推理、多轮对话和复杂推理任务上有优异表现。

FastDeploy 已通过 DeepSeek-v3 的 MoE 实现和 GLM-4.5 的 GQA 注意力积累了丰富的可复用组件，但**线性注意力/Mamba 后端**和**混合注意力调度**是全新模块，需要核心开发。

## 2、功能目标

1. 实现 MiniMax-M1 模型组网代码，提交至 `FastDeploy/fastdeploy/model_executor/models/` 目录
2. 实现 Lightning Attention CUDA 内核（从 vLLM Triton → CUDA C++ 翻译），提交至 `FastDeploy/custom_ops/gpu_ops/` 目录
3. 实现 `MambaBackend(AttentionBackend)` Python 后端，集成至注意力后端框架
4. 适配 FastDeploy 现有的低bit量化推理能力
5. 提交模型使用说明文档

## 3、意义

- 填补 FastDeploy 在混合注意力（标准 + 线性/Mamba）架构的空白
- `MambaBackend` 后端可复用于后续 Mamba 系列模型（如 Jamba、Zamba 等）
- 验证 FastDeploy 注意力后端框架对非 Transformer-only 架构的扩展性

# 二、飞桨现状

## 1、已有基础设施

| 组件 | 文件 | 与 MiniMax-M1 的关系 |
|------|------|---------------------|
| GQA 注意力 | `layers/attention/flash_attn_backend.py` | 直接复用（标准注意力层） |
| MoE 分派 | `layers/moe/fused_moe_*.py` | 直接复用（Top-2 MoE 路由） |
| SharedExpert | `layers/moe/fused_moe_*.py` | 需确认是否支持 shared 路径 |
| Partial RoPE | `layers/rotary_embedding.py` | 直接复用 |
| RMSNorm / DeepNorm | `layers/normalization.py` | RMSNorm 可复用，DeepNorm alpha/beta 缩放需新增 |
| 并行线性层 | `layers/linear.py` | 直接复用 |
| 模型注册 | `models/__init__.py` | 直接复用 |

## 2、缺失组件

| 组件 | 说明 | 复杂度 |
|------|------|--------|
| Lightning Attention CUDA 内核 | 5 个 Triton kernel → CUDA C++ 翻译 | **高** — 核心开发 |
| MambaBackend | 新增注意力后端 | **中** — 遵循 AttentionBackend 接口 |
| 混合注意力调度 | 逐层选择 GQA 或 Mamba | **低** — 配置驱动 |
| Mamba SSM 状态管理 | Block Manager 中管理 Mamba 状态缓存 | **中** — 需扩展 |

# 三、业内方案调研

## 1、vLLM

vLLM 的 MiniMax-M1 支持（`vllm/model_executor/models/minimax_text_01.py`）包含：
- `MiniMaxText01LinearAttention`：使用 Triton 实现的 Lightning Attention（5 个 `@triton.jit` kernel）
- `MiniMaxText01Attention`：标准 GQA 注意力
- 混合层选择通过 `config.attn_type_list` 逐层配置
- MoE + SharedExpert 集成
- DeepNorm 通过 alpha/beta 参数缩放 residual

**关键代码引用**：`vllm/model_executor/layers/lightning_attn.py` 中定义了 5 个核心 Triton kernel：
1. `_fwd_diag_kernel` — 对角块注意力计算
2. `_fwd_kv_parallel` — KV 并行前缀计算
3. `_fwd_kv_reduce` — KV 归约
4. `_fwd_none_diag_kernel` — 非对角块注意力
5. `_linear_attn_decode_kernel` — 解码阶段线性注意力

## 2、对比分析

| 方案 | 优点 | 缺点 |
|------|------|------|
| vLLM (Triton) | 成熟实现，社区验证 | Triton kernel 不可直接用于 Paddle |
| **FastDeploy（本方案）** | 原生 CUDA C++，与 Paddle 深度集成 | CUDA 翻译工作量大，需逐核验证 |

# 四、设计思路与实现方案

## 1、主体设计

**核心策略**：将 vLLM 的 5 个 Triton kernel 翻译为 CUDA C++ 算子，封装为 `MambaBackend(AttentionBackend)`，然后在 MiniMax-M1 模型文件中通过 `attn_type_list` 配置逐层选择 GQA 或 Mamba 后端。

### 架构总览

```
minimax_m1.py (模型定义)
├── MiniMaxM1ForCausalLM(ModelForCasualLM)
│   └── MiniMaxM1Model
│       └── MiniMaxM1DecoderLayer × 80
│           ├── [attn_type=1] StandardAttention → FlashAttnBackend (已有)
│           ├── [attn_type=0] LinearAttention → MambaBackend (新增)
│           ├── MoE + SharedExpert (已有)
│           └── DeepNorm residual scaling (新增逻辑)
│
mamba_backend.py (新增后端)
├── MambaBackend(AttentionBackend)
│   ├── init_attention_metadata()
│   ├── forward_extend() — Prefill 阶段
│   └── forward_decode() — Decode 阶段
│
custom_ops/gpu_ops/mamba_attn/ (新增 CUDA 算子)
├── mamba_impl.cuh — 5 个翻译后的 CUDA kernel
├── mamba.cu — Host 启动器 + 算子注册
└── 注册至 cpp_extensions.cc
```

## 2、关键技术点

### 2.1 Lightning Attention CUDA 翻译

5 个 Triton kernel 的 CUDA C++ 翻译规则：

| Triton 概念 | CUDA 对应 |
|-------------|-----------|
| `tl.program_id(0/1)` | `blockIdx.x / blockIdx.y` |
| `tl.load(ptr + offsets, mask)` | `if (idx < N) data[idx]` |
| `tl.dot(a, b)` | 寄存器级矩阵乘或 `wmma` 指令 |
| `tl.sum(x, axis)` | Warp/Block 归约 |
| `tl.exp(x)` | `expf(x)` |

**验证策略**：每个 kernel 翻译后独立对比 Triton 版本输出，要求 `max_abs_diff < 1e-3`（FP16）。

### 2.2 Mamba SSM 状态管理

与标准 KV Cache 不同，Mamba 的线性注意力维护一个**累积 SSM 状态**（形状 `[batch, heads, head_dim, head_dim]`），在时序步之间传递。

管理方案：
- 在 `ForwardMeta.caches` 中为 Mamba 层分配独立的状态缓存槽位
- Prefill 阶段：从零初始化 SSM 状态，算子内部逐块累积
- Decode 阶段：从缓存读取上一步状态，更新后写回

### 2.3 混合注意力调度

```python
# minimax_m1.py
class MiniMaxM1DecoderLayer:
    def __init__(self, fd_config, layer_id):
        attn_type = fd_config.model_config.attn_type_list[layer_id]
        if attn_type == 0:  # 线性注意力
            self.self_attn = MiniMaxM1LinearAttention(fd_config, layer_id)
        else:  # 标准全注意力
            self.self_attn = MiniMaxM1StandardAttention(fd_config, layer_id)
```

### 2.4 DeepNorm

MiniMax-M1 使用 DeepNorm 变体，对注意力和 MLP 的输出分别应用不同的 alpha/beta 缩放：

```python
# 注意力 residual
hidden_states = residual * alpha_attn + attn_output * beta_attn
# MLP residual
hidden_states = residual * alpha_mlp + mlp_output * beta_mlp
```

alpha/beta 值从 HuggingFace 配置的 `layernorm_full_attention_alpha`、`layernorm_linear_attention_alpha` 等字段读取。

### 2.5 文件结构

| 路径 | 说明 |
|------|------|
| `custom_ops/gpu_ops/mamba_attn/mamba_impl.cuh` | CUDA kernel 实现 |
| `custom_ops/gpu_ops/mamba_attn/mamba.cu` | Host 启动器 + 算子注册 |
| `custom_ops/gpu_ops/cpp_extensions.cc` | 新增 mamba 算子绑定 |
| `fastdeploy/model_executor/layers/attention/mamba_backend.py` | MambaBackend 后端 |
| `fastdeploy/model_executor/layers/attention/ops/mamba_attention.py` | Python 算子包装 |
| `fastdeploy/model_executor/models/minimax_m1.py` | 模型组网代码 |

# 五、测试和验收的考量

1. **CUDA 内核精度**：每个翻译后的 kernel 与 vLLM Triton 版本对比，FP16 最大绝对差 < 1e-3
2. **端到端推理**：对比 HuggingFace transformers 输出，token-level 一致性
3. **量化精度**：WINT8 精度损失 <1%，WINT4 精度损失 <3%
4. **长上下文**：验证 4K/8K/16K 上下文长度下的推理正确性
5. **性能基线**：多卡 A100 首 token 延迟和吞吐量，对标 vLLM

# 六、影响面

## 对用户的影响
- 新增模型选项，无 breaking changes

## 对框架架构的影响
- 新增 `MambaBackend` 注意力后端（扩展点，不修改已有后端）
- 新增 CUDA 算子 `mamba_attention_forward`
- `config.py` 新增 Mamba/DeepNorm 相关配置项

## 对性能的影响
- 不影响现有模型性能
- Mamba 线性注意力的 O(n) 复杂度可显著降低长文本推理延迟

# 七、排期规划

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| Phase 0 | 项目搭建 + 配置解析 + 目录结构 | 1-2天 |
| Phase 1 | Lightning Attention CUDA 内核翻译（5 个 kernel） | 2-3周 |
| Phase 2 | MambaBackend Python 后端 + 算子包装 | 1周 |
| Phase 3 | 模型组网 + 权重加载 + 混合调度 | 1周 |
| Phase 4 | 端到端测试 + 性能调优 + 文档 | 1周 |
| **合计** | | **~5-6周** |

# 名词解释

| 名词 | 说明 |
|------|------|
| Lightning Attention | MiniMax 提出的线性注意力机制，O(n) 时间复杂度 |
| Mamba/SSM | 选择性状态空间模型，一种替代 Transformer 的序列建模方法 |
| DeepNorm | 深层残差连接的归一化策略，用不同 alpha/beta 缩放防止梯度消失 |
| MoE | 混合专家模型，每个 token 路由到 Top-K 专家处理 |

# 附件及参考资料

1. MiniMax-M1 HuggingFace: https://huggingface.co/MiniMaxAI/MiniMax-M1-80B
2. vLLM MiniMax-M1 实现: `vllm/model_executor/models/minimax_text_01.py`
3. vLLM Lightning Attention: `vllm/model_executor/layers/lightning_attn.py`
4. FastDeploy 注意力后端基类: `fastdeploy/model_executor/layers/attention/base_attention_backend.py`
5. 已合并的 H9 MiniMax-M1 RFC: https://github.com/PaddlePaddle/community/pull/1156
6. FastDeploy DeepSeek-v3 MoE 参考: `fastdeploy/model_executor/models/deepseek_v3.py`
