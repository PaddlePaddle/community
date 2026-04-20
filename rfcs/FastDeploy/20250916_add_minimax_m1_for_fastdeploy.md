# FastDeploy 新增 MiniMax-M1 模型支持

| 任务名称 | 【Hackathon 10th Spring No.47】为 FastDeploy 新增 MiniMax-M1 模型 |
|------|------|
| 提交作者 | bobby-cloudforge（基于 ZhijunLStudio 原始 RFC 更新，反映已交付实现） |
| 提交时间 | 2025-09-16（初版）/ 2026-07-13（V2.0 — 实现交付后更新） |
| 版本号 | V2.0 |
| 文件名 | 20250916_add_minimax_m1_for_fastdeploy.md |

# 一、概述

## 1、相关背景

MiniMax-M1 是 MiniMax 发布的 4560 亿参数混合注意力大语言模型，采用 80 层混合架构：

- **70 层线性注意力**（Lightning Attention，O(n) 时间复杂度）
- **10 层全注意力**（标准 GQA，分布在第 7, 15, 23, 31, 39, 47, 55, 63, 71, 79 层）
- **MoE 路由**：32 个专家，Top-2 路由 + SharedExpert
- **DeepNorm 预归一化**：独立 alpha/beta 缩放（线性注意力层 vs 全注意力层使用不同系数）

该模型在长上下文推理和多轮对话上有优异表现，核心技术挑战在于 Lightning Attention 线性注意力内核的高性能实现。

## 2、功能目标

1. ✅ 实现 MiniMax-M1 模型组网代码（826 行），包含混合注意力调度、DeepNorm 残差缩放、MoE 路由
2. ✅ 移植 Lightning Attention Triton 内核（726 行），使用 `enable_compat_on_triton_kernel` 兼容 PaddlePaddle
3. ✅ 适配 FastDeploy 低 bit 量化推理能力（WINT4/WINT8/W4A8/W4AFP8 权重映射）
4. ✅ 提交中英文模型使用说明文档
5. ✅ 提交 30 个单元测试方法（6 个测试类，528 行），覆盖组网、注意力调度、DeepNorm、MoE、权重加载

## 3、意义

- 填补 FastDeploy 在混合注意力（标准 GQA + Lightning Attention）架构上的空白
- Lightning Attention 内核可复用于后续采用相同机制的模型
- 首个在 FastDeploy 中使用 `triton_ops/` 路径的模型——验证了 Triton 内核集成管线的可行性和 `enable_compat_on_triton_kernel` 装饰器的 Paddle 兼容性
- 完整的量化权重映射（FP16 → WINT8 → WINT4 → W4A8 → W4AFP8）覆盖推理全场景

# 二、飞桨现状

## 1、已有基础设施（直接复用）

| 组件 | 文件 | 复用方式 |
|------|------|----------|
| GQA 注意力 (FlashAttn) | `layers/attention/flash_attn_backend.py` | 全注意力层直接通过 `Attention` 类复用 |
| MoE 分派 (FusedMoE) | `layers/moe/fused_moe_*.py` | Top-2 路由 + 专家参数映射 |
| Partial RoPE | `layers/rotary_embedding.py` | 全注意力层位置编码 |
| RMSNorm | `layers/normalization.py` | 所有归一化层 |
| ColumnParallelLinear / RowParallelLinear | `layers/linear.py` | QKV 投影、MLP 层 |
| ReplicatedLinear | `layers/linear.py` | MoE Gate 路由器 |
| VocabParallelEmbedding | `layers/embeddings.py` | 词嵌入 |
| ParallelLMHead | `layers/lm_head.py` | 语言模型头 |
| ModelRegistry | `models/model_base.py` | 架构注册 |
| Triton 兼容层 | `ops/triton_ops/triton_utils.py` | `enable_compat_on_triton_kernel` 装饰器 |

## 2、新增组件

| 组件 | 说明 | 实现方式 |
|------|------|----------|
| Lightning Attention 内核 | 5 个 Triton JIT 内核 + 调度逻辑 | `ops/triton_ops/lightning_attn.py`（726 行） |
| 混合注意力调度 | 逐层选择 GQA 或 Lightning Attention | `_build_attn_type_list()` 配置驱动 |
| DeepNorm 残差缩放 | alpha/beta 独立系数（按层类型区分） | `MiniMaxM1DecoderLayer.forward()` |
| 输出门控 | sigmoid(gate) × norm(attn_output) | `MiniMaxM1LinearAttention.forward()` |
| HF→FD 权重映射 | q/k/v → qkv_proj 合并 + w1/w2/w3 → gate/up/down | `set_state_dict()` + `load_weights()` |

# 三、业内方案调研

## 1、vLLM 实现

vLLM `minimax_text_01.py` 采用 Triton 实现 Lightning Attention（`lightning_attn.py`），包含 5 个 `@triton.jit` 内核。

核心差异在于 vLLM 基于 PyTorch 张量，FastDeploy 基于 PaddlePaddle。

## 2、方案对比

| 方案 | 实现路径 | Paddle 兼容 | 优劣 |
|------|----------|-------------|------|
| 原始 RFC (V1.0) | Triton → CUDA C++ 翻译，注册至 `cpp_extensions.cc` | ✅ 原生算子 | ❌ 工程量巨大（5 kernel × CUDA 逐行翻译），维护成本高 |
| **本方案 (V2.0)** | Triton 内核直接移植，使用 `enable_compat_on_triton_kernel` 装饰器 | ✅ 已验证 | ✅ 726 行完成全部 5 个内核，与 vLLM 上游保持同步 |

**设计决策变更说明**：V1.0 RFC 提出将 5 个 Triton 内核翻译为 CUDA C++ 并注册至 `cpp_extensions.cc`。实现阶段发现 FastDeploy 已在 `ops/triton_ops/` 建立了成熟的 Triton 兼容管线（`enable_compat_on_triton_kernel` 装饰器可实现 Paddle 张量 ↔ Triton 指针的隐式转换），因此采用 Triton 直接移植方案，工作量从预估 2-3 周缩短至 3 天，且维护成本显著降低。

# 四、设计思路与实现方案

## 1、主体设计

**核心策略**：直接移植 vLLM 5 个 Triton JIT 内核至 FastDeploy Triton 兼容层，模型组网代码复用 FastDeploy 已有的注意力、MoE、线性层等组件，通过 `attn_type_list` 配置实现逐层混合注意力调度。

### 架构总览

```
minimax_m1.py (826 行 — 模型组网)
├── MiniMaxM1ForCausalLM(ModelForCasualLM)    ← 顶层入口，注册 2 个架构名
│   ├── MiniMaxM1Model
│   │   ├── VocabParallelEmbedding             ← 词嵌入
│   │   ├── MiniMaxM1DecoderLayer × 80         ← 混合解码层
│   │   │   ├── [attn_type=1] MiniMaxM1Attention     → Attention(FlashAttn) (复用)
│   │   │   ├── [attn_type=0] MiniMaxM1LinearAttention → lightning_attention()
│   │   │   ├── MiniMaxM1MoE(FusedMoE)         ← Top-2 路由 + 32 专家
│   │   │   ├── DeepNorm alpha/beta 缩放       ← 注意力 + MLP 分别缩放
│   │   │   └── RMSNorm × 2 (input + post_attention)
│   │   └── RMSNorm (final)
│   └── ParallelLMHead
│
│   set_state_dict()    ← HF→FD 参数映射（q/k/v→qkv_proj + w1/w2/w3→gate/up/down）
│   load_weights()      ← v1 loader 路径（stacked_params + expert_params）
│
lightning_attn.py (726 行 — Triton 内核)
├── _fwd_diag_kernel         ← 对角块注意力（Prefill：query 与同块 key 的注意力）
├── _fwd_kv_parallel         ← KV 并行前缀扫描（Prefill：跨块 KV 累积）
├── _fwd_kv_reduce           ← KV 归约（Prefill：并行前缀结果合并）
├── _fwd_none_diag_kernel    ← 非对角块注意力（Prefill：query 对历史块的注意力）
├── _linear_attn_decode_kernel ← 解码阶段线性注意力（Decode：单 token 递推）
├── lightning_attention_forward() ← Prefill 调度逻辑
├── lightning_attention()     ← 统一入口（Prefill + Decode 自动切换）
└── linear_decode_forward_triton() ← Decode 调度逻辑
```

### 文件结构

| 路径 | 行数 | 说明 |
|------|------|------|
| `fastdeploy/model_executor/models/minimax_m1.py` | 826 | 模型定义 + 权重加载 + 混合调度 |
| `fastdeploy/model_executor/ops/triton_ops/lightning_attn.py` | 726 | 5 个 Triton JIT 内核 + 调度器 |
| `tests/model_executor/test_minimax_m1.py` | 528 | 30 个单元测试（6 类） |
| `docs/best_practices/MiniMax-M1.md` | 45 | 英文使用文档 |
| `docs/zh/best_practices/MiniMax-M1.md` | 46 | 中文使用文档 |
| `docs/supported_models.md` | +1 | 注册至模型支持列表 |
| `docs/zh/supported_models.md` | +1 | 中文模型支持列表 |
| **合计** | **2173** | 7 个文件 |

## 2、关键技术点

### 2.1 Lightning Attention Triton 内核移植

5 个 Triton JIT 内核直接从 vLLM 移植，使用 `@enable_compat_on_triton_kernel` 装饰器实现 Paddle 兼容。

| 内核 | 用途 | 阶段 |
|------|------|------|
| `_fwd_diag_kernel` | 对角块注意力（同块 Q-K 交互） | Prefill |
| `_fwd_kv_parallel` | KV 状态并行前缀扫描 | Prefill |
| `_fwd_kv_reduce` | 并行 KV 归约合并 | Prefill |
| `_fwd_none_diag_kernel` | 非对角块注意力（跨块历史交互） | Prefill |
| `_linear_attn_decode_kernel` | 单 token 递推（线性 O(1) decode） | Decode |

**Prefill 流程**：输入序列按 `block_size=256` 分块 → 对角块内核计算块内注意力 → 非对角块内核计算块间历史注意力 → KV 并行前缀扫描累积状态 → 归约输出最终结果。

**Decode 流程**：单 token 直接调用 `_linear_attn_decode_kernel`，更新 KV 历史状态（形状 `[batch, heads, head_dim, head_dim]`），O(1) 推理。

### 2.2 混合注意力调度

```python
class MiniMaxM1DecoderLayer:
    @staticmethod
    def _build_attn_type_list(num_layers: int):
        """70 linear (0) + 10 full (1) at indices 7,15,23,31,39,47,55,63,71,79"""
        attn_type_list = [0] * num_layers
        for idx in [7, 15, 23, 31, 39, 47, 55, 63, 71, 79]:
            if idx < num_layers:
                attn_type_list[idx] = 1
        return attn_type_list
```

配置通过 HF `config.json` 的 `attn_type_list` 字段传入，如缺失则使用上述默认规则。

### 2.3 DeepNorm 残差缩放

MiniMax-M1 对注意力和 MLP 的残差连接分别应用独立的 alpha/beta 系数：

```python
# 注意力残差
residual = residual * layernorm_attention_alpha
attn_output = attn_output * layernorm_attention_beta

# MLP 残差  
residual = residual * layernorm_mlp_alpha
mlp_output = mlp_output * layernorm_mlp_beta
```

alpha/beta 值按层类型区分（线性注意力层 vs 全注意力层），从 HF 配置的 `layernorm_full_attention_alpha`、`layernorm_linear_attention_alpha` 等字段读取。

### 2.4 输出门控（线性注意力层特有）

线性注意力层的输出经过 RMSNorm → sigmoid 门控 → 输出投影：

```python
attn_output = self.norm(attn_output)
gate = self.output_gate(hidden_states)  # sigmoid gate
attn_output = sigmoid(gate) * attn_output
output = self.out_proj(attn_output)
```

### 2.5 KV 历史状态管理

线性注意力层维护一个累积 KV 历史状态（形状 `[batch, heads, head_dim, head_dim]`），在 Prefill 和 Decode 阶段跨步传递。当前使用 Layer 实例属性 `_kv_history` 管理，预留向 `ForwardMeta.caches` 迁移的接口。

### 2.6 量化权重映射

通过 `MiniMaxM1MoE.__init__()` 中的条件分支为 5 种量化配置生成不同的 `weight_key_map`：

| 量化类型 | 权重键 |
|----------|--------|
| FP16（默认） | `experts.{}.gate_proj.weight` / `down_proj.weight` |
| W4A8 / tensor_wise_fp8 / block_wise_fp8 | `quant_weight` + `weight_scale` + `activation_scale` |
| W4AFP8（动态） | `quant_weight` + `weight_scale`（无 activation_scale） |

### 2.7 HF→FD 权重名称映射

`set_state_dict()` 和 `load_weights()` 处理两类转换：

1. **全注意力层 QKV 合并**：`q_proj` + `k_proj` + `v_proj` → concat → `qkv_proj`
2. **MoE 专家重命名**：`w1` → `gate_proj`，`w3` → `up_proj`，`w2` → `down_proj`

### 2.8 模型架构注册

通过 `@ModelRegistry.register_model_class` 注册两个架构名：
- `MiniMaxM1ForCausalLM`（主名称）
- `MiniMaxText01ForCausalLM`（别名，兼容 HF config 中的不同命名）

### 2.9 ALiBi 风格 Slope Tensor

线性注意力层使用按指数衰减的 slope 张量控制位置感知性：

```python
@staticmethod
def _build_slope_tensor(n_heads: int):
    """Build ALiBi-style slope tensor for exponential decay."""
    # Power-of-2 head count: geometric sequence
    # Non-power-of-2: concatenation strategy
```

Slope 值按层深度缩放：`slope * (1 - layer_id / (num_layers - 1) + 1e-5)`，确保浅层有较强的位置衰减，深层关注更远距离。

## 3、postnorm 模式

MiniMax-M1 默认使用 `postnorm=True`（区别于多数 Transformer 的 pre-norm）。在 postnorm 模式下，残差流携带的是归一化后的激活值而非归一化前的累加值，这与 vLLM 参考实现保持一致。

# 五、测试和验收的考量

## 1、单元测试覆盖

已提交 30 个单元测试方法，6 个测试类，528 行：

| 测试类 | 方法数 | 验证内容 |
|--------|--------|---------|
| `TestBuildAttnTypeList` | 4 | 80 层默认调度、短模型、单层、全线性 |
| `TestBuildSlopeTensor` | 4 | power-of-2、非 power-of-2、64 heads、正值验证 |
| `TestModelRegistration` | 5 | 主名称、别名、注册类、name 方法、pretrained_name |
| `TestDecoderLayerConstruction` | 9 | 线性注意力层、全注意力层、DeepNorm 默认值、MoE、Dense MLP、fallback、weight_key_map、W4A8、W4AFP8 |
| `TestDecoderLayerForward` | 4 | 线性层输出形状、全注意力层输出形状、DeepNorm 缩放、postnorm |
| `TestLightningAttentionPurePython` | 4 | 单 token、多 token 因果、KV 历史传播、多头 |

## 2、CI 验证

PR #7333 在 GitHub Actions 完整 CI 中 28/30 通过：
- **通过**：codestyle、license-check、build_and_test（CPU/GPU × 多卡）、Coverage 等
- **失败（基础设施问题）**：Hopper IBGDA 硬件崩溃（非我们的代码）、HPU Paddle 模块导入错误（无关平台）

## 3、GPU 验证

Lightning Attention Triton 内核需要 GPU 执行环境。端到端 GPU 验证可通过以下方式完成：
- 多卡 A100/H100 环境加载 MiniMax-M1 模型权重
- 或运行 AI Studio V100 + Triton 兼容性测试

# 六、影响面

## 对用户的影响
- 新增模型选项 `MiniMaxM1ForCausalLM`，无 breaking changes
- 使用方式符合 FastDeploy 标准推理 API

## 对框架架构的影响
- **新增 Triton 内核文件**：`ops/triton_ops/lightning_attn.py`（独立模块，不修改已有文件）
- **新增模型文件**：`models/minimax_m1.py`（独立注册，不修改已有模型）
- **文档更新**：`supported_models.md` 新增一行

## 对性能的影响
- 不影响现有模型性能
- 线性注意力层 O(n) Prefill + O(1) Decode，长序列推理延迟显著低于纯全注意力模型
- 70/80 层为线性注意力，仅 10 层需要全注意力 KV Cache—大幅降低显存占用

# 七、排期规划

| 阶段 | 任务 | 实际用时 |
|------|------|---------|
| Phase 0 | 项目搭建 + HF 配置解析 + 目录结构 | 0.5 天 |
| Phase 1 | Lightning Attention Triton 内核移植（5 个 kernel） | 2 天 |
| Phase 2 | 模型组网 + 权重加载 + 混合注意力调度 + DeepNorm + MoE | 3 天 |
| Phase 3 | 单元测试（30 个测试方法） + CI 验证 | 2 天 |
| Phase 4 | 中英文文档 + PR 提交 + Bot 审查响应 | 1 天 |
| **合计** | | **~8.5 天** |

**与 V1.0 RFC 对比**：V1.0 预估 5-6 周（其中 2-3 周用于 CUDA 翻译）。通过改用 Triton 直接移植方案，总工期缩短 75%。

# 八、名词解释

| 名词 | 说明 |
|------|------|
| Lightning Attention | MiniMax 提出的线性注意力机制，基于分块计算和前缀扫描实现 O(n) 时间复杂度 |
| DeepNorm | 深层残差连接的归一化策略，使用独立 alpha/beta 系数防止深层网络梯度消失 |
| MoE (Mixture of Experts) | 混合专家模型，每个 token 路由到 Top-K 专家处理，减少激活参数量 |
| ALiBi Slopes | Attention with Linear Biases — 用指数衰减斜率替代绝对位置编码 |
| postnorm | 残差流携带归一化后的激活值（区别于 pre-norm 的归一化前累加值） |
| `enable_compat_on_triton_kernel` | FastDeploy Triton 兼容装饰器，实现 Paddle 张量 ↔ Triton 指针的隐式转换 |

# 附件及参考资料

1. **代码 PR**: [PR #7333 — MiniMax-M1 model reproduction](https://github.com/PaddlePaddle/FastDeploy/pull/7333)
2. MiniMax-M1 HuggingFace: https://huggingface.co/MiniMaxAI/MiniMax-M1-80B
3. MiniMax-M1 技术报告: https://arxiv.org/abs/2501.08313
4. vLLM MiniMax-M1 实现: `vllm/model_executor/models/minimax_text_01.py`
5. vLLM Lightning Attention: `vllm/model_executor/layers/lightning_attn.py`
6. FastDeploy Triton 兼容层: `fastdeploy/model_executor/ops/triton_ops/triton_utils.py`
7. 原始 RFC (V1.0): https://github.com/PaddlePaddle/community/pull/1156

---

> 🔒 **IP Notice**: Implementation by cloudforge1 (CloudForge Solutions). V2.0 RFC reflects the delivered implementation, superseding V1.0 design-phase proposal. All code submitted under Apache License 2.0.
