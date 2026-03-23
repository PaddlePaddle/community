# FastDeploy 新增 SD3 / Flux 扩散模型推理支持

| 任务名称 | 【Hackathon 10th Spring No.48】为 FastDeploy 新增 SD3 / Flux 扩散模型 |
|------|------|
| 提交作者 | cloudforge1 |
| 提交时间 | 2026-07-08 |
| 版本号 | V1.0 |
| 文件名 | 20260708_add_sd3_flux_diffusion_models_for_fastdeploy.md |

# 一、概述

## 1、相关背景

Stable Diffusion 3 (SD3) 和 Flux 是当前主流的文生图扩散模型：
- **SD3** (Stability AI)：采用 MMDiT (Multi-Modal Diffusion Transformer) 架构，独立处理文本和图像 token 后进行 joint attention
- **Flux** (Black Forest Labs)：基于 SD3 演进，使用 Double Stream + Single Stream Transformer 块的混合架构

FastDeploy 当前专注于 LLM 推理部署，**不包含扩散模型推理管线**。新增 SD3/Flux 支持需要从零搭建扩散模型的推理引擎、调度器和组件框架，属于全新功能模块。

## 2、功能目标

1. 在 FastDeploy 中新增 `diffusion_models/` 目录，提供扩散模型推理框架
2. 实现 SD3 (MMDiT) 和 Flux (Double/Single Stream) 的推理组网
3. 集成 VAE 编解码器、文本编码器（CLIP + T5）、采样调度器
4. 提供 OpenAI Images API 兼容的 HTTP 接口
5. 支持 FP16/BF16 推理和基本量化

## 3、意义

- 将 FastDeploy 从纯 LLM 推理引擎扩展为**多模态推理平台**
- `DiffusionEngine` 框架可复用于未来扩散模型（SDXL、Kolors、Hunyuan 等）
- 填补飞桨生态在扩散模型高性能部署的空白

# 二、飞桨现状

## 1、飞桨生态中的扩散模型支持

| 项目 | 支持情况 |
|------|---------|
| PaddleMIX | 已支持 SD/SDXL 训练和推理（PaddlePaddle 原生） |
| PPDiffusers | 飞桨版 Diffusers，支持 SD1.5/SDXL/SD3 |
| FastDeploy | **不支持**扩散模型 — 仅有 LLM 推理管线 |

## 2、FastDeploy 可复用组件

| 组件 | 现有支持 | 扩散模型适配 |
|------|---------|-------------|
| HTTP 服务框架 | `entrypoints/openai/` | 新增 `/v1/images/generations` 路由 |
| 张量并行 | `layers/linear.py` | DiT 中的线性层可复用 |
| 量化框架 | `layers/quantization/` | DiT 量化需适配 |
| 配置管理 | `config.py` | 新增 `DiffusionConfig` 类 |

## 3、缺失组件

| 组件 | 说明 | 复杂度 |
|------|------|--------|
| DiffusionEngine | 扩散模型推理编排引擎 | **高** — 全新模块 |
| DiT 模型组网 | MMDiT (SD3) / DoubleStream+SingleStream (Flux) | **高** — 核心开发 |
| VAE 编解码 | 图像潜空间编解码 | **中** — 参考 PPDiffusers |
| 文本编码器 | CLIP-L/CLIP-G + T5-XXL 编码管线 | **中** — 参考 PPDiffusers |
| 采样调度器 | Flow Matching / Euler 调度器 | **低** — 算法明确 |
| Images API | OpenAI `/v1/images/generations` 接口 | **低** — 接口定义明确 |

# 三、业内方案调研

## 1、ComfyUI

- 基于节点的扩散模型工作流引擎
- SD3/Flux 原生支持，高度模块化
- 不适合生产部署（无 API 化、无张量并行）

## 2、SGLang

- `sglang` 项目在 v0.4+ 版本初步支持 DiT 模型推理
- 采用 `DiTManager` 调度，与 LLM 的 `RadixAttention` 分离
- 使用 PyTorch 后端

## 3、Diffusers (HuggingFace)

- Python 扩散模型库，SD3/Flux 原生支持
- 提供完整的管线抽象：`scheduler` + `unet/dit` + `vae` + `text_encoder`
- 参考价值最高：管线组织方式、调度器接口设计

## 4、对比分析

| 方案 | 优点 | 缺点 |
|------|------|------|
| ComfyUI | 灵活工作流 | 不可 API 化部署 |
| SGLang | DiT 调度优化 | PyTorch 后端，不支持 Paddle |
| Diffusers | 完整管线抽象 | Python 推理，无部署优化 |
| **FastDeploy（本方案）** | PaddlePaddle 原生，API 化，可与 LLM 共存 | 开发量大 |

# 四、设计思路与实现方案

## 1、整体架构

采用**管线式架构**，与 FastDeploy 现有的 LLM 引擎并列而非嵌套：

```
fastdeploy/
├── model_executor/          # 现有 LLM 推理
│   ├── models/              # LLM 模型
│   └── layers/              # LLM 层
│
├── diffusion_models/        # 新增：扩散模型推理 (与 model_executor 并列)
│   ├── __init__.py
│   ├── engine.py            # DiffusionEngine 入口
│   ├── config.py            # DiffusionConfig
│   ├── schedulers/          # 采样调度器
│   │   ├── base.py
│   │   ├── euler.py
│   │   └── flow_matching.py
│   ├── components/          # 共享组件
│   │   ├── vae.py           # VAE 编解码器
│   │   ├── text_encoder.py  # CLIP + T5 编码管线
│   │   └── timestep_embedding.py
│   ├── models/              # DiT 模型定义
│   │   ├── sd3_dit.py       # SD3 MMDiT
│   │   └── flux_dit.py      # Flux Double/Single Stream
│   └── api/                 # Images API
│       └── images.py        # /v1/images/generations
│
├── entrypoints/openai/
│   ├── api_server.py        # 修改：添加 /v1/images 路由
│   └── ...
```

## 2、关键技术点

### 2.1 DiffusionEngine

扩散推理的核心编排器，管理去噪循环：

```python
class DiffusionEngine:
    def __init__(self, config: DiffusionConfig):
        self.dit = load_dit_model(config)        # SD3 或 Flux
        self.vae = load_vae(config)              # AutoencoderKL
        self.text_encoder = TextEncoderPipeline(config)  # CLIP + T5
        self.scheduler = create_scheduler(config) # Flow Matching / Euler

    def generate(self, prompt, num_steps=28, guidance_scale=7.0, ...):
        # 1. 文本编码
        text_embeddings = self.text_encoder.encode(prompt)
        # 2. 初始化潜空间噪声
        latents = paddle.randn([1, 16, H//8, W//8])
        # 3. 去噪循环
        for t in self.scheduler.timesteps(num_steps):
            noise_pred = self.dit(latents, t, text_embeddings)
            latents = self.scheduler.step(noise_pred, t, latents)
        # 4. VAE 解码
        image = self.vae.decode(latents)
        return image
```

### 2.2 SD3 MMDiT

SD3 的核心是 Multi-Modal Diffusion Transformer：
- **输入**: 图像 token（潜空间 patch）+ 文本 token（CLIP+T5 编码）
- **结构**: N 个 JointTransformerBlock，每个块包含独立的图像/文本 AdaLN + QKV 投影 + Joint Attention + 独立 MLP
- Joint Attention: 图像和文本 QKV concat 后做统一 attention，再 split 回去

```python
class SD3JointTransformerBlock(paddle.nn.Layer):
    def forward(self, img_hidden, txt_hidden, timestep_emb):
        # AdaLN modulation (独立)
        img_mod = self.img_adaln(timestep_emb)
        txt_mod = self.txt_adaln(timestep_emb)
        # QKV projection (独立)
        img_qkv = self.img_attn_qkv(img_hidden * img_mod.scale + img_mod.shift)
        txt_qkv = self.txt_attn_qkv(txt_hidden * txt_mod.scale + txt_mod.shift)
        # Joint attention
        q = concat([img_qkv.q, txt_qkv.q])
        k = concat([img_qkv.k, txt_qkv.k])
        v = concat([img_qkv.v, txt_qkv.v])
        attn_out = scaled_dot_product_attention(q, k, v)
        img_attn, txt_attn = split(attn_out)
        # MLP (独立)
        img_hidden = img_hidden + img_attn + self.img_mlp(img_hidden)
        txt_hidden = txt_hidden + txt_attn + self.txt_mlp(txt_hidden)
        return img_hidden, txt_hidden
```

### 2.3 Flux Double/Single Stream

Flux 使用两段式架构：
1. **Double Stream 块**: 类似 SD3 JointTransformerBlock，但 attention 计算后图像和文本独立更新
2. **Single Stream 块**: 图像和文本 concat 为统一序列，用标准 Transformer 块处理

```python
class FluxModel(paddle.nn.Layer):
    def __init__(self, config):
        self.double_blocks = nn.LayerList([
            FluxDoubleStreamBlock(config) for _ in range(config.num_double_blocks)
        ])
        self.single_blocks = nn.LayerList([
            FluxSingleStreamBlock(config) for _ in range(config.num_single_blocks)
        ])

    def forward(self, img_hidden, txt_hidden, timestep_emb, rotary_emb):
        # Double stream phase
        for block in self.double_blocks:
            img_hidden, txt_hidden = block(img_hidden, txt_hidden, timestep_emb, rotary_emb)
        # Merge streams
        hidden = concat([txt_hidden, img_hidden], axis=1)
        # Single stream phase
        for block in self.single_blocks:
            hidden = block(hidden, timestep_emb, rotary_emb)
        # Extract image output
        img_hidden = hidden[:, txt_hidden.shape[1]:]
        return img_hidden
```

### 2.4 文本编码器管线

SD3/Flux 使用 3 个文本编码器的组合：

| 编码器 | 模型 | 输出维度 | 用途 |
|--------|------|---------|------|
| CLIP-L | openai/clip-vit-large-patch14 | 768 | 文本嵌入 (pool) |
| CLIP-G | laion/CLIP-ViT-bigG-14 | 1280 | 文本嵌入 (pool) |
| T5-XXL | google/t5-v1_1-xxl | 4096 | 序列嵌入 (last hidden) |

输出拼接策略：`pooled = concat(clip_l_pool, clip_g_pool)`，`sequence = t5_hidden_states`

### 2.5 实现优先级

鉴于本任务的开发量较大，采用分阶段交付：

| 优先级 | 内容 | 状态 |
|--------|------|------|
| **P0** | DiffusionEngine + FlowMatchingScheduler + Flux DiT | 核心，必须完成 |
| **P0** | VAE 编解码 + CLIP/T5 文本编码 | 核心，必须完成 |
| **P1** | SD3 MMDiT | 高优先，Flux 验证后扩展 |
| **P1** | Images API (`/v1/images/generations`) | 高优先，接口化 |
| **P2** | DiT 量化（FP16→INT8） | 后续优化 |
| **P2** | 张量并行 | 后续优化 |

# 五、测试和验收的考量

1. **端到端生成**：给定固定 prompt + seed，生成图像与 Diffusers 参考实现 pixel-level PSNR > 30dB
2. **模型加载**：从 HuggingFace 权重正确加载 SD3 和 Flux 模型
3. **API 功能**：`/v1/images/generations` 接口返回正确的 base64 图像
4. **性能基线**：单卡 A100 上 512×512 图像生成延迟 < 10s (28 steps)
5. **内存占用**：Flux-schnell (FP16) 单卡 A100 (80GB) 可运行

# 六、影响面

## 对用户的影响
- 新增图像生成能力，不影响现有 LLM 功能
- 通过标准 FastDeploy 接口调用

## 对框架架构的影响
- 新增 `diffusion_models/` 顶层模块（与 `model_executor/` 并列）
- `api_server.py` 添加图像生成路由
- 不修改现有 LLM 推理代码

## 对性能的影响
- 不影响现有 LLM 推理性能
- 扩散模型推理为独立进程/线程

# 七、排期规划

| 阶段 | 任务 | 预计时间 |
|------|------|---------|
| Phase 0 | 框架搭建（目录结构 + DiffusionEngine + Config） | 2-3天 |
| Phase 1 | 组件实现（VAE + CLIP/T5 + Flow Matching 调度器） | 1周 |
| Phase 2 | Flux DiT 模型组网 + 权重加载 | 1-1.5周 |
| Phase 3 | SD3 MMDiT 模型组网 + 权重加载 | 1周 |
| Phase 4 | Images API + 端到端测试 + 文档 | 1周 |
| **合计** | | **~4-5周** |

# 名词解释

| 名词 | 说明 |
|------|------|
| DiT | Diffusion Transformer，用 Transformer 替代 UNet 的扩散模型架构 |
| MMDiT | Multi-Modal DiT，SD3 采用的双流 Joint Attention 架构 |
| Flow Matching | SD3/Flux 使用的连续时间扩散训练/采样方法，替代 DDPM |
| VAE | Variational Autoencoder，图像与潜空间之间的编解码器 |
| AdaLN | Adaptive Layer Normalization，用 timestep 调制归一化参数 |

# 附件及参考资料

1. Stable Diffusion 3 论文: https://arxiv.org/abs/2403.03206
2. Flux 模型: https://huggingface.co/black-forest-labs/FLUX.1-schnell
3. HuggingFace Diffusers SD3 管线: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/
4. PPDiffusers (飞桨版): https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers
5. SGLang DiT 支持: https://github.com/sgl-project/sglang
6. FastDeploy API 服务: `fastdeploy/entrypoints/openai/api_server.py`
