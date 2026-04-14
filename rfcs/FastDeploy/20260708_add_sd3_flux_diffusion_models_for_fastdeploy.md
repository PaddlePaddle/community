# FastDeploy 新增 SD3 / Flux 扩散模型推理支持

| 任务名称 | 【Hackathon 10th Spring No.48】为 FastDeploy 新增 SD3 / Flux 扩散模型 |
|------|------|
| 提交作者 | r-cloudforge |
| 提交时间 | 2026-07-13 |
| 版本号 | V2.0 |
| 文件名 | 20260708_add_sd3_flux_diffusion_models_for_fastdeploy.md |
| 变更说明 | V2.0：修正目录结构至 `model_executor/diffusion_models/`；新增 PR #4021 失败分析；新增 FD 基础设施集成方案；调整为 Flux-first 分阶段策略；补充代码规模估算与 PPDiffusers 复用分析；强化测试方案 |

# 一、概述

## 1、相关背景

Stable Diffusion 3 (SD3) 和 Flux 是当前主流的文生图扩散模型：
- **SD3** (Stability AI)：采用 MMDiT (Multi-Modal Diffusion Transformer) 架构，独立处理文本和图像 token 后进行 joint attention
- **Flux** (Black Forest Labs)：基于 SD3 演进，使用 Double Stream + Single Stream Transformer 块的混合架构

FastDeploy 当前专注于 LLM 推理部署，**不包含扩散模型推理管线**。上游任务要求将扩散模型组网代码提交至 `fastdeploy/model_executor/diffusion_models/` 目录下，作为 `model_executor` 的子模块而非独立顶层模块。这意味着扩散模型需要与现有 LLM 推理框架共享注册、配置、权重加载等基础设施。

## 2、功能目标

1. 在 `fastdeploy/model_executor/diffusion_models/` 下新增扩散模型推理框架
2. **Phase 1（Flux-first）**：实现 Flux (Double/Single Stream) 的完整推理管线
3. **Phase 2（SD3 跟进）**：复用 Phase 1 框架扩展 SD3 (MMDiT) 支持
4. 集成 VAE 编解码器、文本编码器（CLIP + T5）、Flow Matching 采样调度器
5. 与 FD 现有基础设施对接：`ModelCategory` 注册、`FDConfig` 配置、`ImageMediaIO` 图像输出、OpenAI Images API
6. 支持 FP16/BF16 推理和基本量化

## 3、意义

- 将 FastDeploy 从纯 LLM 推理引擎扩展为**多模态推理平台**
- 扩散模型作为 `model_executor` 子模块，天然复用 FD 的模型注册、权重加载、并行推理等能力
- `DiffusionEngine` 框架可复用于未来扩散模型（SDXL、Kolors、Hunyuan 等）
- 填补飞桨企业级部署生态在扩散模型高性能推理的空白——PPDiffusers 面向研究和训练，FD 面向**生产部署**

# 二、飞桨现状

## 1、飞桨生态中的扩散模型支持

| 项目 | 定位 | SD3/Flux 支持 |
|------|------|--------------|
| PaddleMIX / PPDiffusers | 训练+研究推理（Paddle 原生） | ✅ SD3 + Flux 组网 + 调度器 |
| FastDeploy | **企业级高性能部署** | ❌ 仅 LLM 推理管线 |

**关键区分**：PPDiffusers 提供 Paddle 原生的扩散模型组网和调度器代码，可作为参考实现直接复用（零 PyTorch 依赖）。FastDeploy 的价值在于将其封装为**生产级部署服务**——标准 API、模型注册、权重管理、并行推理、量化优化。

## 2、PPDiffusers 可复用代码分析

| PPDiffusers 模块 | 文件位置 | 代码量 | 复用率 | 说明 |
|------------------|---------|--------|--------|------|
| `FluxTransformer2DModel` | `ppdiffusers/models/transformers/flux_transformer.py` | ~773 行 | ~80% | Paddle 原生组网，需适配 FD weight loading |
| `SD3Transformer2DModel` | `ppdiffusers/models/transformers/transformer_sd3.py` | ~500 行 | ~80% | JointTransformerBlock 可直接复用 |
| `AutoencoderKL` | `ppdiffusers/models/autoencoders/autoencoder_kl.py` | ~400 行 | ~85% | VAE 编解码，接口稳定 |
| `FlowMatchEulerDiscreteScheduler` | `ppdiffusers/schedulers/scheduling_euler_discrete.py` | ~200 行 | ~90% | 调度器逻辑简单，可近乎直接使用 |
| CLIP/T5 编码器 | `ppdiffusers/transformers/` | ~600 行 | ~70% | 需适配 FD 文本编码管线 |

**诚实的整体复用评估**：
- 模型 Transformer 组网代码：**~80% 可复用**（已是 PaddlePaddle 原生，零 torch 依赖）
- 集成代码（引擎、配置、注册、API、测试）：**0% 复用，全部新写**
- 综合复用率：**~40%**——组网省力，但集成工作量 ≥ 组网本身

## 3、FastDeploy 可复用组件

| 组件 | 现有位置 | 扩散模型适配方案 |
|------|---------|-----------------|
| 模型注册 | `model_base.py` `ModelCategory` + `LazyRegisteredModel` | 新增 `IMAGE_GENERATION` 类别 |
| HTTP 服务框架 | `entrypoints/openai/api_server.py` | 新增 `/v1/images/generations` 路由 |
| 图像输出编码 | `multimodal/image.py` `ImageMediaIO` | 直接复用 base64/URL 编码 |
| 权重加载 | `model_executor/load_weight_utils.py` | 扩展支持 diffusion checkpoint 格式 |
| 配置管理 | `config.py` `FDConfig` | 新增 `DiffusionConfig` 子配置 |
| 张量并行 | `layers/linear.py` | DiT 中的线性层可复用 |
| 量化框架 | `layers/quantization/` | DiT 量化需适配 |

# 三、业内方案调研

## 1、ComfyUI

- 基于节点的扩散模型工作流引擎
- SD3/Flux 原生支持，高度模块化
- **不适合生产部署**：无标准 API、无张量并行、无权重管理

## 2、SGLang

- v0.4+ 版本初步支持 DiT 模型推理
- 采用 `DiTManager` 调度，与 LLM 的 `RadixAttention` 分离
- 使用 PyTorch 后端

## 3、Diffusers (HuggingFace)

- Python 扩散模型库，SD3/Flux 原生支持
- 提供完整的管线抽象：`scheduler` + `dit` + `vae` + `text_encoder`
- 参考价值最高：管线组织方式、调度器接口设计

## 4、PPDiffusers (飞桨版 Diffusers)

- PaddleMIX 子项目，PaddlePaddle 原生实现
- 完整支持 SD3/Flux 组网和推理
- **核心价值**：**代码已是 Paddle 原生，零 torch 依赖**，组网代码可高比例直接复用
- 定位为研究/训练工具，缺乏企业级部署能力（API 化、模型注册、生产级服务化）

## 5、PR #4021 失败分析

PR [#4021](https://github.com/PaddlePaddle/FastDeploy/pull/4021) 由 @kitalkuyo-gita 提交，约 15,000 行，被 reviewer @chang-wenbin 拒绝。核心问题及本方案的规避策略：

| #4021 失败原因 | 本方案规避策略 |
|---------------|---------------|
| **使用 PyTorch mock imports**（`import torch` 作为依赖） | PPDiffusers 组网代码**原生 PaddlePaddle**，零 torch 依赖。所有 `paddle.nn.Layer`，无需任何 mock |
| **未说明企业级部署收益** | 本 RFC 明确阐述 FD 集成价值：ModelCategory 注册、FDConfig 配置、标准 API、权重管理、并行推理——这些是 PPDiffusers 不具备的 |
| **单 PR 过大（~15K 行）** | 分阶段交付：Phase 1 仅 Flux（~950-1,200 行），每个 PR 聚焦单一功能边界 |

**核心教训**：扩散模型进入 FastDeploy 的价值不在于"又一个推理实现"，而在于**利用 FD 基础设施提供生产级部署能力**。组网代码应从 PPDiffusers 复用（已 Paddle 原生），开发重心放在集成层。

## 6、对比分析

| 方案 | 优点 | 缺点 |
|------|------|------|
| ComfyUI | 灵活工作流 | 不可 API 化部署 |
| SGLang | DiT 调度优化 | PyTorch 后端，不支持 Paddle |
| Diffusers | 完整管线抽象 | Python 推理，无部署优化 |
| PPDiffusers | **Paddle 原生组网**，高复用率 | 研究定位，缺乏部署基础设施 |
| **FastDeploy（本方案）** | Paddle 原生 + 企业级 FD 基础设施 + 标准 API | 集成开发量中等 |

# 四、设计思路与实现方案

## 1、整体架构

按照上游任务要求，扩散模型代码置于 `model_executor/diffusion_models/` **子目录**下，与现有 `models/`、`layers/` 并列，共享 `model_executor` 的注册和加载基础设施：

```
fastdeploy/
├── model_executor/                # 推理执行层
│   ├── models/                    # LLM 模型（现有）
│   ├── layers/                    # 共享层（现有）
│   ├── load_weight_utils.py       # 权重加载（扩展）
│   │
│   └── diffusion_models/          # 新增：扩散模型推理
│       ├── __init__.py
│       ├── engine.py              # DiffusionEngine 推理编排 (~200行)
│       ├── schedulers/
│       │   ├── __init__.py
│       │   └── flow_matching.py   # FlowMatchEulerDiscrete (~120行)
│       ├── components/
│       │   ├── __init__.py
│       │   ├── vae.py             # AutoencoderKL 封装 (~150行)
│       │   └── text_encoder.py    # CLIP + T5 编码管线 (~180行)
│       └── models/
│           ├── __init__.py
│           ├── flux_dit.py        # Flux Double/Single Stream (~350行)
│           └── sd3_dit.py         # SD3 MMDiT (Phase 2, ~300行)
│
├── config.py                      # 修改：新增 DiffusionConfig
├── multimodal/image.py            # 现有：ImageMediaIO（直接复用）
└── entrypoints/openai/
    └── api_server.py              # 修改：新增 /v1/images/generations
```

**代码规模估算（Phase 1 Flux-only）**：
| 文件 | 预估行数 | 性质 |
|------|---------|------|
| `engine.py` | ~200 | **全新** — DiffusionEngine 编排 |
| `flux_dit.py` | ~350 | ~80% 复用 PPDiffusers + FD 适配 |
| `text_encoder.py` | ~180 | ~70% 复用 + FD 编码管线封装 |
| `vae.py` | ~150 | ~85% 复用 AutoencoderKL |
| `flow_matching.py` | ~120 | ~90% 复用调度器 |
| FD 集成代码（config/registration/API） | ~150 | **全新** |
| **Phase 1 合计** | **~950-1,200** | — |

## 2、FastDeploy 基础设施集成方案

### 2.1 模型注册（model_base.py）

在 `ModelCategory` 枚举中新增 `IMAGE_GENERATION` 类别，通过 `LazyRegisteredModel` 注册 Flux/SD3：

```python
# model_base.py — ModelCategory 扩展
class ModelCategory(IntFlag):
    TEXT_GENERATION = auto()
    MULTIMODAL = auto()
    EMBEDDING = auto()
    REASONING = auto()
    REWARD = auto()
    IMAGE_GENERATION = auto()  # 新增

# model_base.py — ModelRegistry._enhanced_models 注册
"FluxTransformer2DModel": {
    "module_name": "flux_dit",
    "module_path": "fastdeploy.model_executor.diffusion_models.models",
    "class_name": "FluxForImageGeneration",
    "category": ModelCategory.IMAGE_GENERATION,
},
```

### 2.2 配置管理（config.py）

在 `FDConfig` 中新增扩散模型相关配置，与现有 LLM 配置并存：

```python
@dataclass
class DiffusionConfig:
    """扩散模型专用配置"""
    num_inference_steps: int = 28
    guidance_scale: float = 7.0
    image_height: int = 512
    image_width: int = 512
    scheduler_type: str = "flow_matching_euler"
    vae_path: Optional[str] = None
    text_encoder_paths: Optional[Dict[str, str]] = None

class FDConfig:
    # ... 现有 LLM 配置 ...
    diffusion: Optional[DiffusionConfig] = None  # 新增
```

### 2.3 权重加载（load_weight_utils.py）

扩展现有权重加载工具支持扩散模型 checkpoint 格式：
- HuggingFace diffusion checkpoint（`model_index.json` + 分目录权重）
- safetensors 格式（SD3/Flux 标准）
- 组件级独立加载（`transformer/`、`vae/`、`text_encoder*/`）

### 2.4 图像输出（multimodal/image.py）

`ImageMediaIO` 已存在于 `fastdeploy/multimodal/image.py`，支持 PIL Image → base64/URL 编码。扩散模型生成的 PIL Image 直接通过此接口返回，**零额外开发**。

### 2.5 API 路由（api_server.py）

在现有 FastAPI 应用中新增 OpenAI Images API 兼容路由：

```python
# entrypoints/openai/api_server.py — 新增路由
@app.post("/v1/images/generations")
async def create_image(request: ImageGenerationRequest):
    """OpenAI Images API 兼容接口"""
    engine = get_diffusion_engine()
    images = await engine.generate(
        prompt=request.prompt,
        n=request.n,
        size=request.size,
    )
    return ImageGenerationResponse(
        data=[ImageMediaIO.encode(img) for img in images]
    )
```

## 3、关键技术方案

### 3.1 DiffusionEngine

扩散推理的核心编排器，管理完整的去噪循环：

```python
class DiffusionEngine:
    def __init__(self, config: DiffusionConfig):
        self.dit = load_dit_model(config)        # Flux 或 SD3
        self.vae = load_vae(config)              # AutoencoderKL
        self.text_encoder = TextEncoderPipeline(config)  # CLIP + T5
        self.scheduler = create_scheduler(config) # Flow Matching Euler

    async def generate(self, prompt, num_steps=28, guidance_scale=7.0, ...):
        # 1. 文本编码
        text_embeddings = self.text_encoder.encode(prompt)
        # 2. 初始化潜空间噪声
        latents = paddle.randn([1, 16, H // 8, W // 8])
        # 3. 去噪循环
        for t in self.scheduler.timesteps(num_steps):
            noise_pred = self.dit(latents, t, text_embeddings)
            latents = self.scheduler.step(noise_pred, t, latents)
        # 4. VAE 解码
        image = self.vae.decode(latents)
        return image
```

### 3.2 Flux Double/Single Stream 模型

Flux 使用两段式架构——前段双流独立处理图像和文本，后段合并为统一序列：

```python
class FluxForImageGeneration(paddle.nn.Layer):
    """Flux DiT — Double Stream + Single Stream 混合架构"""
    def __init__(self, config):
        self.double_blocks = nn.LayerList([
            FluxDoubleStreamBlock(config) for _ in range(config.num_double_blocks)
        ])
        self.single_blocks = nn.LayerList([
            FluxSingleStreamBlock(config) for _ in range(config.num_single_blocks)
        ])

    def forward(self, img_hidden, txt_hidden, timestep_emb, rotary_emb):
        # Double stream: 图像和文本独立处理 + joint attention
        for block in self.double_blocks:
            img_hidden, txt_hidden = block(img_hidden, txt_hidden, timestep_emb, rotary_emb)
        # 合并为统一序列
        hidden = paddle.concat([txt_hidden, img_hidden], axis=1)
        # Single stream: 标准 Transformer 处理
        for block in self.single_blocks:
            hidden = block(hidden, timestep_emb, rotary_emb)
        # 提取图像输出
        img_hidden = hidden[:, txt_hidden.shape[1]:]
        return img_hidden
```

### 3.3 SD3 MMDiT（Phase 2）

SD3 的核心是 Multi-Modal Diffusion Transformer，图像和文本 token 在每层进行 joint attention：

```python
class SD3JointTransformerBlock(paddle.nn.Layer):
    def forward(self, img_hidden, txt_hidden, timestep_emb):
        # AdaLN modulation (独立)
        img_mod = self.img_adaln(timestep_emb)
        txt_mod = self.txt_adaln(timestep_emb)
        # QKV projection (独立) + Joint attention (concat → attn → split)
        q = paddle.concat([img_q, txt_q])
        k = paddle.concat([img_k, txt_k])
        v = paddle.concat([img_v, txt_v])
        attn_out = scaled_dot_product_attention(q, k, v)
        img_attn, txt_attn = paddle.split(attn_out, ...)
        # MLP (独立)
        img_hidden = img_hidden + img_attn + self.img_mlp(...)
        txt_hidden = txt_hidden + txt_attn + self.txt_mlp(...)
        return img_hidden, txt_hidden
```

### 3.4 文本编码器管线

SD3/Flux 使用 3 个文本编码器的组合：

| 编码器 | 模型 | 输出维度 | 用途 |
|--------|------|---------|------|
| CLIP-L | openai/clip-vit-large-patch14 | 768 | 文本嵌入 (pool) |
| CLIP-G | laion/CLIP-ViT-bigG-14 | 1280 | 文本嵌入 (pool) |
| T5-XXL | google/t5-v1_1-xxl | 4096 | 序列嵌入 (last hidden) |

输出拼接策略：`pooled = concat(clip_l_pool, clip_g_pool)`，`sequence = t5_hidden_states`

## 4、实现分阶段策略（Flux-first）

本任务采用 **Flux-first** 分阶段交付策略。Flux 架构相比 SD3 更成熟（Flux-schnell 推理步数低至 4 步，社区活跃度更高），先完成 Flux 可快速验证整体框架的正确性，SD3 随后复用框架扩展。

| 阶段 | 范围 | 交付物 | PR 规模 |
|------|------|--------|---------|
| **Phase 1（Flux-only）** | DiffusionEngine + FlowMatchingScheduler + Flux DiT + VAE + CLIP/T5 + FD 集成 | 完整 Flux 推理管线 | ~950-1,200 行 |
| **Phase 2（SD3 扩展）** | SD3 MMDiT 组网 + SD3-specific 配置 | SD3 模型支持 | ~400-500 行 |
| **Phase 3（优化）** | DiT 量化 (FP16→INT8) + 张量并行 | 性能优化 | 视情况 |

# 五、测试和验收的考量

## 1、单元测试

| 测试类型 | 内容 | 工具 |
|----------|------|------|
| 组网正确性 | 每个 Layer/Block 的输入输出 shape 和 dtype 验证 | `pytest` + `paddle.randn` 构造输入 |
| 调度器正确性 | FlowMatchingEuler 的 timestep 生成和 step 计算，对比 PPDiffusers 参考输出 | `np.testing.assert_allclose(atol=1e-5)` |
| 配置加载 | `DiffusionConfig` 从 JSON/YAML 加载、默认值验证 | `pytest` |
| API 接口 | `/v1/images/generations` 请求响应格式、参数校验 | `httpx.AsyncClient` |

## 2、端到端集成测试

| 测试项 | 验收标准 |
|--------|---------|
| **图像生成一致性** | 给定固定 prompt + seed，生成图像与 PPDiffusers 参考实现 pixel-level PSNR ≥ 30dB |
| **对比 PPDiffusers 推理延迟** | 同等配置下（Flux-schnell, 512×512, 4 steps），FD 推理延迟 ≤ PPDiffusers 的 1.1x（不应因集成引入显著开销） |
| **绝对性能基线** | 单卡 A100 上 512×512 图像生成延迟 < 5s (Flux-schnell 4 steps) / < 15s (Flux-dev 28 steps) |
| **模型加载** | 从 HuggingFace 权重正确加载 Flux 模型，无权重缺失或 shape mismatch |
| **API 功能** | `/v1/images/generations` 接口返回正确的 base64 编码图像，响应格式兼容 OpenAI Images API |
| **内存占用** | Flux-schnell (FP16) 单卡 A100 (80GB) 可运行，peak memory < 40GB |

## 3、测试策略

- **Phase 1**：单元测试 + Flux 端到端测试，确保框架可用
- **Phase 2**：SD3 端到端测试复用 Phase 1 测试框架，增加 MMDiT 特有的 joint attention 正确性验证
- 测试代码提交至 `FastDeploy/tests/model_executor/diffusion_models/`

# 六、影响面

## 对用户的影响
- 新增图像生成能力，不影响现有 LLM 功能
- 通过标准 FastDeploy `/v1/images/generations` API 调用

## 对框架架构的影响
- `model_executor/` 下新增 `diffusion_models/` **子目录**
- `model_base.py` 新增 `IMAGE_GENERATION` ModelCategory 枚举值
- `config.py` 新增 `DiffusionConfig` 配置类
- `load_weight_utils.py` 扩展支持 diffusion checkpoint 格式
- `api_server.py` 添加图像生成路由
- **不修改现有 LLM 推理代码逻辑**

## 对性能的影响
- 不影响现有 LLM 推理性能
- 扩散模型推理为独立引擎实例

## 对已有测试的影响
- 不修改已有测试——新增测试目录 `tests/model_executor/diffusion_models/`

# 七、排期规划

| 阶段 | 任务 | 预计时间 | 产出 |
|------|------|---------|------|
| Phase 0 | FD 集成层（ModelCategory + DiffusionConfig + API 路由注册） | 2-3天 | 框架骨架，可 import 但不可运行 |
| Phase 1a | Flux DiT 组网 + FlowMatchingScheduler（复用 PPDiffusers） | 1周 | 模型可 forward |
| Phase 1b | VAE + CLIP/T5 文本编码器 + DiffusionEngine 编排 | 1周 | Flux 端到端生成 |
| Phase 1c | 测试 + 调优 + PR 提交 | 3-5天 | **Phase 1 PR（Flux-only, ~1,000行）** |
| Phase 2 | SD3 MMDiT 组网 + 测试 | 1-1.5周 | **Phase 2 PR（SD3, ~500行）** |
| Phase 3 | 量化 + 张量并行（可选） | 1周 | 性能优化 PR |
| **合计** | | **~5-6周** | |

# 名词解释

| 名词 | 说明 |
|------|------|
| DiT | Diffusion Transformer，用 Transformer 替代 UNet 的扩散模型架构 |
| MMDiT | Multi-Modal DiT，SD3 采用的双流 Joint Attention 架构 |
| Flow Matching | SD3/Flux 使用的连续时间扩散训练/采样方法，替代 DDPM |
| VAE | Variational Autoencoder，图像与潜空间之间的编解码器 |
| AdaLN | Adaptive Layer Normalization，用 timestep 调制归一化参数 |
| PPDiffusers | PaddleMIX 子项目，飞桨版 Diffusers，Paddle 原生扩散模型库 |
| Double Stream | Flux 前段架构：图像和文本分别处理后做 joint attention |
| Single Stream | Flux 后段架构：图像和文本合并为统一序列用标准 Transformer 处理 |

# 附件及参考资料

1. Stable Diffusion 3 论文: https://arxiv.org/abs/2403.03206
2. Flux 模型: https://huggingface.co/black-forest-labs/FLUX.1-schnell
3. HuggingFace Diffusers SD3 管线: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/
4. PPDiffusers (飞桨版): https://github.com/PaddlePaddle/PaddleMIX/tree/develop/ppdiffusers
5. PPDiffusers Flux Transformer: `ppdiffusers/models/transformers/flux_transformer.py`
6. PPDiffusers SD3 Transformer: `ppdiffusers/models/transformers/transformer_sd3.py`
7. SGLang DiT 支持: https://github.com/sgl-project/sglang
8. FastDeploy 模型注册: `fastdeploy/model_executor/models/model_base.py` — `ModelCategory`, `LazyRegisteredModel`
9. FastDeploy API 服务: `fastdeploy/entrypoints/openai/api_server.py`
10. FastDeploy 图像输出: `fastdeploy/multimodal/image.py` — `ImageMediaIO`
11. PR #4021 (rejected): https://github.com/PaddlePaddle/FastDeploy/pull/4021 — PyTorch mock 依赖、单 PR 过大