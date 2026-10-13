# FastDeploy 新增 SD3、Flux 扩散模型支持设计文档

| 任务名称 | 为 FastDeploy 新增 SD3、Flux 扩散模型 |
|------|------|
| 提交作者 | xiejunlin |
| 提交时间 | 2026-03-17 |
| 版本号 | V1.0 |
| 文件名 | 20260317_add_sd3_flux_diffusion_models_for_fastdeploy.md |

# 一、概述

## 1、相关背景

Stable Diffusion 3（SD3）和 Flux.1 是当前最先进的文生图扩散模型，均采用 Diffusion Transformer（DiT）架构替代了传统的 UNet 骨干网络，在图像生成质量和文本理解能力上取得了显著突破。

- **SD3** 由 Stability AI 发布，基于 MMDiT（Multi-Modal Diffusion Transformer）架构，使用三个文本编码器（CLIP-L + OpenCLIP-G + T5-XXL）和 16 通道 VAE，参数量从 2B（Medium）到 8B（Large）。
- **Flux.1** 由 Black Forest Labs 发布，采用混合 Transformer 架构（19 层 Double Stream + 38 层 Single Stream），使用两个文本编码器（CLIP-L + T5-XXL），总参数量约 12B。Flux.1 有三个变体：schnell（1-4 步快速生成）、dev（20-50 步高质量）、pro（API 服务）。

FastDeploy 当前专注于 LLM/VLM 推理服务，尚未支持扩散模型。PaddleMIX 项目中的 PPDiffusers 已有 SD3 和 Flux 的 PaddlePaddle 实现，但缺乏高性能推理服务能力。本任务旨在为 FastDeploy 引入扩散模型推理和服务能力，填补这一空白。

## 2、功能目标

* 在 FastDeploy 中实现 SD3 和 Flux 扩散模型的高性能推理
* 提供 OpenAI Images API 兼容的服务接口（`/v1/images/generations`）
* 适配 FastDeploy 现有的量化推理能力（FP8、WINT4、WINT8 等）
* 支持张量并行（Tensor Parallelism）实现多卡推理
* 提供 CLI 入口和离线推理接口

## 3、意义

扩展 FastDeploy 从纯文本生成到多模态生成的能力边界，为用户提供统一的大模型推理服务平台，覆盖 LLM、VLM 和图像生成三大场景。

# 二、现状分析

## FastDeploy 现状

FastDeploy 当前的架构围绕 LLM 推理设计，核心执行流为 **Engine → Worker → ModelRunner → Model**，存在以下与扩散模型不兼容的设计假设：

| 维度 | LLM 推理（当前） | 扩散模型推理（目标） |
|------|----------------|-------------------|
| 执行模式 | 逐 token 自回归生成 | 固定步数迭代去噪 |
| 缓存机制 | KV Cache 管理 | 无 KV Cache，需管理 latent 状态 |
| 输出格式 | token 序列/文本 | 图像张量/base64 编码 |
| 调度策略 | prefill-decode 两阶段 | 批量请求并行去噪 |
| 模型基类 | `ModelForCasualLM`（logits 输出） | 需要新基类（noise prediction 输出） |
| API 接口 | Chat/Completion | Images Generation |

## 可复用的基础设施

| 组件 | 复用方式 |
|------|---------|
| 量化框架（FP8、WINT4/8 等） | 直接应用于 Transformer 和 VAE 的线性层 |
| 张量并行线性层 | 用于 MMDiT/Flux Transformer 的 QKV 投影和 MLP |
| 注意力后端（Flash Attention 等） | 用于 Joint Attention 计算 |
| RMSNorm / LayerNorm | 用于 Transformer 块的归一化 |
| RoPE 实现 | Flux 使用 2D RoPE 编码空间位置 |
| 分布式通信 | 张量并行的 all-reduce 通信 |
| Worker 抽象 | GPU/XPU 等多硬件支持 |
| FastAPI 服务框架 | 扩展新的 API 端点 |

## PaddleMIX/PPDiffusers 现有实现

PaddleMIX 项目中 PPDiffusers 已有完整的 Paddle 实现，可作为组网参考：
- `ppdiffusers/models/transformer_sd3.py` — SD3Transformer2DModel（MMDiT）
- `ppdiffusers/pipelines/flux/modeling_flux.py` — FluxTransformer2DModel
- `ppdiffusers/models/autoencoder_kl.py` — AutoencoderKL（16 通道 VAE）
- `ppdiffusers/schedulers/scheduling_flow_match_euler_discrete.py` — FlowMatchEulerDiscreteScheduler

# 三、业内方案调研

## 1) HuggingFace Diffusers

业内最成熟的扩散模型推理框架，提供完整的 SD3 和 Flux pipeline 实现。核心设计：
- Pipeline 模式：将文本编码、去噪循环、VAE 解码封装为统一 pipeline
- 组件可替换：scheduler、text encoder、transformer、VAE 均可独立替换
- 优化手段：`torch.compile`、Flash Attention、VAE tiling、model offloading
- 局限：面向单用户推理，缺乏高并发服务能力和请求调度

## 2) ComfyUI / AUTOMATIC1111

社区主流的扩散模型推理 UI 工具，以节点图/WebUI 形式提供推理能力。优化手段包括模型缓存、注意力切片、FP8 量化等。但本质是单机单用户工具，不具备生产级服务能力。

## 3) TensorRT / ONNX Runtime

NVIDIA TensorRT 提供了 SD3 的图优化和算子融合加速方案，可实现 2-4x 推理加速。但需要预编译引擎，灵活性较差，且不提供服务框架。

## 4) SGLang / vLLM / vLLM-Omni

SGLang 近期开始探索扩散模型支持（DiT serving），采用类似 LLM 的请求调度方式管理扩散模型推理。

vLLM 主线目前不支持扩散模型，但 vLLM 社区于 2025 年 11 月发布了 **vLLM-Omni**（`vllm-project/vllm-omni`），作为 vLLM 的官方扩展，将支持范围从自回归文本模型扩展到全模态模型（文本、图像、视频、音频的生成）。其核心设计：

- **Stage 抽象**: 将复杂的多模态模型分解为有向图中的多个 Stage（如文本编码器 → DiT 去噪 → VAE 解码），每个 Stage 拥有独立的调度器、显存管理和执行引擎，可针对不同计算模式（自回归 vs 迭代去噪）分别优化。
- **分离式执行后端**: 各 Stage 可运行在不同的硬件资源上并独立扩缩容，实现动态资源分配。
- **Diffusion Cache 加速**: 支持 Cache-DiT、TeaCache 等缓存加速方法，通过跨去噪步骤复用中间计算结果来减少冗余计算，在几乎不损失质量的前提下加速推理。
- **API 设计**: 通过 `--output-modalities image` 参数同时暴露 `/v1/chat/completions`（内联 base64 图像）和 `/v1/images/generations` 端点。

参考文献：[Fully Disaggregated Serving for Any-to-Any Multimodal Models (arXiv 2602.02204)](https://arxiv.org/abs/2602.02204)

vLLM-Omni 的 Stage 分解模式和分离式调度为 FastDeploy 的扩散模型集成提供了有价值的参考方向。

## 调研结论

当前业内的扩散模型部署方案中，vLLM-Omni 通过 Stage 分解和分离式调度提供了较为先进的架构设计，但其基于 PyTorch 生态。FastDeploy 已有的 PaddlePaddle 量化、并行、服务基础设施为在飞桨生态中实现同等能力提供了良好基础。

# 四、设计思路与实现方案

## 总体架构

在 FastDeploy 现有架构基础上，新增扩散模型推理路径：

```
用户请求
  │
  ▼
API Server ──── /v1/images/generations (新增)
  │
  ▼
DiffusionEngine (新增，管理扩散推理生命周期)
  │
  ▼
GpuWorker (复用) ──── DiffusionModelRunner (新增)
  │
  ▼
DiffusionModel (新增基类)
  ├── TextEncoder (CLIP / T5)
  ├── Transformer (SD3 MMDiT / Flux Hybrid)
  ├── VAE Decoder
  └── Scheduler (FlowMatchEuler)
```

## 目录结构

```
fastdeploy/model_executor/diffusion_models/    # 扩散模型主目录（新增）
├── __init__.py
├── diffusion_base.py                          # 扩散模型基类
├── sd3/                                       # SD3 模型
│   ├── __init__.py
│   ├── sd3_transformer.py                     # MMDiT Transformer
│   └── sd3_pipeline.py                        # SD3 推理 pipeline
├── flux/                                      # Flux 模型
│   ├── __init__.py
│   ├── flux_transformer.py                    # Hybrid Transformer
│   └── flux_pipeline.py                       # Flux 推理 pipeline
├── components/                                # 共享组件
│   ├── __init__.py
│   ├── vae.py                                 # AutoencoderKL (16通道)
│   ├── text_encoders.py                       # CLIP / T5 文本编码器
│   ├── scheduler.py                           # FlowMatchEulerDiscreteScheduler
│   └── embeddings.py                          # 时间步/位置编码
├── layers/                                    # 扩散模型专用层
│   ├── __init__.py
│   ├── dit_attention.py                       # Joint Attention / DiT Attention
│   ├── dit_block.py                           # MMDiT Block / Single-Double Stream Block
│   └── adaptive_norm.py                       # adaLN-Zero / adaLN-Single
fastdeploy/engine/diffusion_engine.py          # 扩散推理引擎（新增）
fastdeploy/worker/diffusion_model_runner.py    # 扩散模型 Runner（新增）
fastdeploy/entrypoints/openai/serving_images.py # 图像生成 API（新增）
```

如需开发自定义算子：
```
custom_ops/gpu_ops/
└── diffusion_ops/                             # 扩散模型专用算子（按需）
    ├── fused_adaln.cu                         # 融合 adaLN 算子
    └── patchify.cu                            # 融合 patchify/unpatchify 算子
```

<!-- PLACEHOLDER_PHASE1 -->

---

### Phase 1: 基础框架搭建 (1 周)

**目标**: 建立扩散模型的基类、配置系统和推理引擎骨架。

* **新增** `fastdeploy/model_executor/diffusion_models/diffusion_base.py`：定义文生图推理的统一接口（如 `encode_prompt`, `denoise_step`, `generate` 等）。
* **修改** `fastdeploy/config.py`：
* 新增 `DiffusionConfig` 数据类，包含步数、引导强度、宽高、VAE Tiling 等配置。
* 在 `FDConfig` 中支持 `runner_type = "diffusion"`。


* **新增** `fastdeploy/engine/diffusion_engine.py`：实现轻量级引擎，负责管理扩散推理的生命周期。
* **新增** `fastdeploy/worker/diffusion_model_runner.py`：继承 `ModelRunnerBase`，负责根据模型类型加载对应的 Pipeline。

**里程碑**: 完成基础框架后，具备扩散模型的加载和空跑能力。


### Phase 2: 共享组件实现 (1 周)

**目标**: 实现 SD3 和 Flux 共享的核心组件——文本编码器、VAE、调度器。

* **新增** `components/text_encoders.py`：实现 CLIP-L、OpenCLIP-G 和 T5-XXL 文本编码器组。
* **新增** `components/vae.py`：实现 16 通道 VAE 编解码器，支持 VAE Tiling 优化。
* **新增** `components/scheduler.py`：实现 `FlowMatchEulerDiscreteScheduler` 调度器。
* **新增** `components/embeddings.py`：实现时间步编码、Patch 嵌入以及 Flux 专用的 2D RoPE。

**里程碑**: 完成共享组件后，文本编码、VAE 解码、调度器均可独立测试验证。

### Phase 3: SD3 Transformer 组网 & Phase 4: Flux Transformer 组网

**目标**: 实现 SD3 的 MMDiT（Multi-Modal Diffusion Transformer）骨干网络和完整推理 pipeline。实现 Flux 的混合 Transformer 架构（Double Stream + Single Stream）和完整推理 pipeline。

* **新增** `layers/adaptive_norm.py`：实现 adaLN-Zero 和 adaLN-Single 自适应归一化层。
* **新增** `layers/dit_attention.py`：实现 Joint Self-Attention（联合注意力机制）。
* **新增** `layers/dit_block.py`：实现 MMDiT Block（SD3）以及 Double/Single Stream Block（Flux）。
* **新增** `sd3/` 目录：包含 SD3 特有的 Transformer 组网和推理 Pipeline。
* **新增** `flux/` 目录：包含 Flux 特有的混合 Transformer 组网和推理 Pipeline。

**里程碑**: SD3 模型可加载权重并完成端到端文生图推理。Flux 模型可加载权重并完成端到端文生图推理，支持 schnell 和 dev 两个变体。

### Phase 5: API 服务与 CLI 集成 (0.5 周)

**目标**: 提供 OpenAI Images API 兼容的服务接口和 CLI 入口。

* **修改** `fastdeploy/entrypoints/openai/protocol.py`：新增 `ImageGenerationRequest` 和 `ImageGenerationResponse` 协议类。
* **新增** `fastdeploy/entrypoints/openai/serving_images.py`：实现符合 OpenAI 标准的 `/v1/images/generations` 服务接口。
* **修改** `fastdeploy/entrypoints/openai/api_server.py`：注册图像生成路由。
* **修改** `fastdeploy/entrypoints/cli/main.py`：新增 `run_diffusion` 子命令支持离线推理。

**里程碑**: 用户可通过 CLI 启动扩散模型服务，并通过 OpenAI 兼容 API 调用图像生成。

### Phase 6: 量化与并行适配 (1 周)

**目标**: 适配 FastDeploy 现有的量化和张量并行能力，降低显存占用并支持多卡推理。

* **修改** `layers/dit_attention.py` 与 `dit_block.py`：将标准线性层替换为 `get_quantized_linear`，以支持 FP8、WINT4/8 量化。
* **新增** `custom_ops/gpu_ops/diffusion_ops/`：实现高性能 C++/CUDA 融合算子（如 `fused_adaln`）。

**里程碑**: Flux 12B 可在单卡 24GB GPU 上以 FP8 运行，或在 2 卡上以 FP16 张量并行运行。

# 五、测试和验收的考量

## 1、单元测试

```
tests/diffusion_models/
├── test_vae.py                    # VAE 编解码正确性
├── test_scheduler.py              # FlowMatch 调度器步进正确性
├── test_text_encoders.py          # 文本编码输出形状和数值
├── test_sd3_transformer.py        # SD3 MMDiT 前向输出形状
├── test_flux_transformer.py       # Flux Transformer 前向输出形状
├── test_dit_blocks.py             # MMDiT Block / Single-Double Stream Block
└── test_quantization.py           # 量化后模型输出一致性
```

## 2、端到端测试

- SD3 Medium 文生图：加载权重，生成 1024x1024 图像，验证输出为合理图像（非噪声）
- Flux schnell 文生图：4 步生成，验证输出质量
- Flux dev 文生图：28 步生成，与 PPDiffusers 参考输出对比 PSNR/SSIM
- 量化推理：FP8/WINT8 量化后生成图像，与 FP16 基线对比质量

## 3、API 测试

- `/v1/images/generations` 端点：请求/响应格式兼容性
- 并发请求：多请求批量处理正确性
- 错误处理：无效参数、超时等边界情况

## 4、性能基准

| 指标 | SD3 Medium | Flux schnell | Flux dev |
|------|-----------|-------------|---------|
| 单图延迟 (1024x1024) | 目标 < 5s | 目标 < 3s | 目标 < 10s |
| 吞吐量 (images/min) | 待测 | 待测 | 待测 |
| 显存占用 (FP16) | < 12GB | < 24GB | < 24GB |
| 显存占用 (FP8) | < 8GB | < 14GB | < 14GB |

# 六、可行性分析

## 可行性分析

1. **组网参考充分**: PPDiffusers 已有完整的 Paddle 实现，可直接参考模型结构和权重映射
2. **基础设施成熟**: FastDeploy 的量化、并行、服务框架可直接复用
3. **架构兼容**: 扩散模型的 Transformer 结构与 LLM 共享大量基础组件（线性层、注意力、归一化）
4. **风险可控**: 扩散模型推理流程相对固定（编码→去噪→解码），不涉及复杂的调度策略

# 七、影响面

## 新增模块

本任务为纯新增功能，不修改 FastDeploy 现有 LLM 推理逻辑：

- 新增 `fastdeploy/model_executor/diffusion_models/` 目录（扩散模型组网代码）
- 新增 `fastdeploy/engine/diffusion_engine.py`（扩散推理引擎）
- 新增 `fastdeploy/worker/diffusion_model_runner.py`（扩散模型 Runner）
- 新增 `fastdeploy/entrypoints/openai/serving_images.py`（图像生成 API）
- 可选新增 `custom_ops/gpu_ops/diffusion_ops/`（融合算子）

## 需修改的现有文件

| 文件 | 修改内容 |
|------|---------|
| `fastdeploy/config.py` | 新增 `DiffusionConfig`，`runner_type` 支持 `"diffusion"` |
| `fastdeploy/entrypoints/openai/api_server.py` | 注册 `/v1/images/generations` 路由 |
| `fastdeploy/entrypoints/openai/protocol.py` | 新增 `ImageGenerationRequest/Response` |
| `fastdeploy/entrypoints/cli/main.py` | 新增 `run_diffusion` 子命令 |

## 文档交付

- `docs/get_started/sd3_flux.md` — SD3/Flux 部署指南
- `docs/features/image_generation.md` — 图像生成 API 使用说明
- `docs/quantization/diffusion_quant.md` — 扩散模型量化指南
